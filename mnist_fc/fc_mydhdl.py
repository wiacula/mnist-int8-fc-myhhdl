from dataclasses import dataclass

import numpy as np
from myhdl import Signal, always, always_comb, always_seq, block, enum, intbv

from .fc_reference import QuantizedFullyConnected


FIXED_SHIFT = 40
FIXED_HALF = 1 << (FIXED_SHIFT - 1)
ACCUMULATOR_MIN = -(1 << 31)
ACCUMULATOR_MAX = 1 << 31


@dataclass(frozen=True)
class HardwareConstants:
    weights: tuple[int, ...]
    matmul_multipliers: tuple[int, ...]
    add_multiplier: int
    add_bias_offsets: tuple[int, ...]
    matmul_output_zero_point: int
    output_zero_point: int
    input_size: int
    output_size: int

    @classmethod
    def from_layer(cls, layer: QuantizedFullyConnected) -> "HardwareConstants":
        if layer.input_zero_point != 0:
            raise ValueError("The MyHDL core currently requires an input zero point of 0")
        if np.any(layer.weight_zero_points != 0):
            raise ValueError("The MyHDL core currently requires weight zero points of 0")

        shift_factor = 1 << FIXED_SHIFT
        matmul_scales = (
            np.float32(layer.input_scale)
            * layer.weight_scales
            / np.float32(layer.matmul_output_scale)
        )
        matmul_multipliers = tuple(
            int(np.rint(float(scale) * shift_factor)) for scale in matmul_scales
        )
        add_matmul_scale = np.float32(layer.matmul_output_scale) / np.float32(
            layer.output_scale
        )
        add_bias_scale = np.float32(layer.bias_scale) / np.float32(layer.output_scale)
        add_multiplier = int(np.rint(float(add_matmul_scale) * shift_factor))
        bias_multiplier = int(np.rint(float(add_bias_scale) * shift_factor))
        bias_values = layer.bias.reshape(layer.output_size).astype(np.int64)
        add_bias_offsets = tuple(
            int(bias_value - layer.bias_zero_point) * bias_multiplier
            for bias_value in bias_values
        )
        accumulator_bound = max(
            sum(abs(int(weight)) for weight in layer.weights[:, output_index]) * 255
            for output_index in range(layer.output_size)
        )
        if accumulator_bound > (1 << 31) - 1:
            raise ValueError("The model's fully connected accumulator may exceed int32 range")
        weights = tuple(
            int(layer.weights[input_index, output_index])
            for output_index in range(layer.output_size)
            for input_index in range(layer.input_size)
        )
        return cls(
            weights=weights,
            matmul_multipliers=matmul_multipliers,
            add_multiplier=add_multiplier,
            add_bias_offsets=add_bias_offsets,
            matmul_output_zero_point=layer.matmul_output_zero_point,
            output_zero_point=layer.output_zero_point,
            input_size=layer.input_size,
            output_size=layer.output_size,
        )


@block
def fc_serial(
    clk,
    reset,
    input_data,
    input_valid,
    input_ready,
    output_data,
    output_valid,
    output_ready,
    weights: tuple[int, ...],
    matmul_multipliers: tuple[int, ...],
    add_multiplier: int,
    add_bias_offsets: tuple[int, ...],
    matmul_output_zero_point: int,
    output_zero_point: int,
    input_size: int,
    output_size: int,
):
    states = enum("LOAD", "MAC", "MAC_FLUSH", "REQUANTIZE", "ADD_BIAS", "OUTPUT")
    state = Signal(states.LOAD)
    input_memory = [
        Signal(intbv(0, min=-256, max=256))
        for _ in range(input_size)
    ]
    even_weight_memory = [
        Signal(
            intbv(
                weights[(output_index * 2) * input_size + input_index],
                min=-128,
                max=128,
            )
        )
        for output_index in range((output_size + 1) // 2)
        for input_index in range(input_size)
    ]
    odd_weight_memory = [
        Signal(
            intbv(
                weights[(output_index * 2 + 1) * input_size + input_index],
                min=-128,
                max=128,
            )
        )
        for output_index in range(output_size // 2)
        for input_index in range(input_size)
    ]
    matmul_multiplier_memory = [
        Signal(intbv(multiplier, min=-(1 << 31), max=(1 << 31)))
        for multiplier in matmul_multipliers
    ]
    add_bias_offset_memory = [
        Signal(intbv(offset, min=-(1 << 47), max=(1 << 47)))
        for offset in add_bias_offsets
    ]
    input_load_index = Signal(intbv(0, min=0, max=input_size))
    input_index = Signal(intbv(0, min=0, max=input_size))
    output_index = Signal(intbv(0, min=0, max=output_size))
    weight_bank_address = Signal(
        intbv(0, min=0, max=((output_size + 1) // 2) * input_size)
    )
    input_value = Signal(intbv(0, min=-256, max=256))
    weight_value = Signal(intbv(0, min=-128, max=128))
    even_weight_value = Signal(intbv(0, min=-128, max=128))
    odd_weight_value = Signal(intbv(0, min=-128, max=128))
    input_value_extended = Signal(intbv(0, min=-(1 << 15), max=(1 << 15)))
    weight_value_extended = Signal(intbv(0, min=-(1 << 15), max=(1 << 15)))
    mac_product = Signal(intbv(0, min=-(1 << 15), max=(1 << 15)))
    accumulator = Signal(intbv(0, min=ACCUMULATOR_MIN, max=ACCUMULATOR_MAX))
    matmul_multiplier = Signal(intbv(0, min=-(1 << 31), max=(1 << 31)))
    add_bias_offset = Signal(intbv(0, min=-(1 << 47), max=(1 << 47)))
    matmul_accumulator_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    matmul_multiplier_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    matmul_product = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    matmul_value_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    add_centered_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    add_multiplier_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    add_offset_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    add_product = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    mac_primed = Signal(bool(0))
    matmul_value = Signal(intbv(0)[8:])
    output_quantized = Signal(intbv(0)[8:])
    output_register = Signal(intbv(0)[8:])

    @always_comb
    def control_outputs():
        input_ready.next = state == states.LOAD and not reset
        output_valid.next = state == states.OUTPUT and not reset
        output_data.next = output_register

    @always(clk.posedge)
    def read_memories():
        input_value.next = input_memory[input_index]

    @always(clk.posedge)
    def read_weight_banks():
        even_weight_value.next = even_weight_memory[weight_bank_address]
        odd_weight_value.next = odd_weight_memory[weight_bank_address]

    @always_comb
    def select_weight_bank():
        if output_index % 2 == 0:
            weight_value.next = even_weight_value
        else:
            weight_value.next = odd_weight_value

    @always_comb
    def read_weight():
        weight_bank_address.next = (output_index // 2) * input_size + input_index

    @always_comb
    def select_requantization_constants():
        matmul_multiplier.next = matmul_multiplier_memory[output_index]
        add_bias_offset.next = add_bias_offset_memory[output_index]

    @always_comb
    def extend_matmul_operands():
        matmul_accumulator_extended.next = accumulator
        matmul_multiplier_extended.next = matmul_multiplier

    @always_comb
    def multiply_matmul():
        matmul_product.next = matmul_accumulator_extended * matmul_multiplier_extended

    @always(clk.posedge)
    def write_input_memory():
        if not reset and state == states.LOAD and input_valid:
            input_memory[input_load_index].next = input_data

    @always_comb
    def extend_mac_operands():
        input_value_extended.next = input_value
        weight_value_extended.next = weight_value

    @always_comb
    def multiply_mac():
        mac_product.next = input_value_extended * weight_value_extended

    @always_comb
    def extend_matmul_value():
        matmul_value_extended.next = matmul_value

    @always_comb
    def center_matmul_value():
        add_centered_extended.next = matmul_value_extended - matmul_output_zero_point

    @always_comb
    def extend_add_operands():
        add_multiplier_extended.next = add_multiplier
        add_offset_extended.next = add_bias_offset

    @always_comb
    def multiply_add():
        add_product.next = (
            add_centered_extended * add_multiplier_extended
            + add_offset_extended
        )

    @always_comb
    def requantize_matmul():
        quotient = matmul_product >> FIXED_SHIFT
        rounded = quotient
        if matmul_product[FIXED_SHIFT:0] > FIXED_HALF or (
            matmul_product[FIXED_SHIFT:0] == FIXED_HALF and (quotient & 1) != 0
        ):
            rounded = quotient + 1
        quantized = rounded + matmul_output_zero_point
        if quantized < 0:
            matmul_value.next = 0
        elif quantized > 255:
            matmul_value.next = 255
        else:
            matmul_value.next = quantized

    @always_comb
    def requantize_bias_add():
        quotient = add_product >> FIXED_SHIFT
        rounded = quotient
        if add_product[FIXED_SHIFT:0] > FIXED_HALF or (
            add_product[FIXED_SHIFT:0] == FIXED_HALF and (quotient & 1) != 0
        ):
            rounded = quotient + 1
        quantized = rounded + output_zero_point
        if quantized < 0:
            output_quantized.next = 0
        elif quantized > 255:
            output_quantized.next = 255
        else:
            output_quantized.next = quantized

    @always_seq(clk.posedge, reset=reset)
    def sequencer():
        if reset:
            state.next = states.LOAD
            input_load_index.next = 0
            input_index.next = 0
            output_index.next = 0
            accumulator.next = 0
            mac_primed.next = False
            output_register.next = 0
        elif state == states.LOAD:
            if input_valid:
                if input_load_index == input_size - 1:
                    input_load_index.next = 0
                    input_index.next = 0
                    output_index.next = 0
                    accumulator.next = 0
                    mac_primed.next = False
                    state.next = states.MAC
                else:
                    input_load_index.next = input_load_index + 1
        elif state == states.MAC:
            if not mac_primed:
                mac_primed.next = True
                input_index.next = 1
            else:
                accumulator.next = accumulator + mac_product
                if input_index == input_size - 1:
                    state.next = states.MAC_FLUSH
                else:
                    input_index.next = input_index + 1
        elif state == states.MAC_FLUSH:
            accumulator.next = accumulator + mac_product
            mac_primed.next = False
            state.next = states.REQUANTIZE
        elif state == states.REQUANTIZE:
            state.next = states.ADD_BIAS
        elif state == states.ADD_BIAS:
            output_register.next = output_quantized
            state.next = states.OUTPUT
        elif state == states.OUTPUT:
            if output_ready:
                if output_index == output_size - 1:
                    input_load_index.next = 0
                    state.next = states.LOAD
                else:
                    output_index.next = output_index + 1
                    input_index.next = 0
                    accumulator.next = 0
                    mac_primed.next = False
                    state.next = states.MAC

    return (
        control_outputs,
        read_memories,
        read_weight_banks,
        select_weight_bank,
        read_weight,
        select_requantization_constants,
        extend_mac_operands,
        multiply_mac,
        extend_matmul_operands,
        multiply_matmul,
        write_input_memory,
        extend_matmul_value,
        center_matmul_value,
        extend_add_operands,
        multiply_add,
        requantize_matmul,
        requantize_bias_add,
        sequencer,
    )


@block
def fc_parallel2(
    clk,
    reset,
    input_data,
    input_valid,
    input_ready,
    output_data,
    output_valid,
    output_ready,
    weights: tuple[int, ...],
    matmul_multipliers: tuple[int, ...],
    add_multiplier: int,
    add_bias_offsets: tuple[int, ...],
    matmul_output_zero_point: int,
    output_zero_point: int,
    input_size: int,
    output_size: int,
):
    if output_size % 2:
        raise ValueError("The two-MAC design requires an even output size")

    states = enum("LOAD", "MAC", "MAC_FLUSH", "REQUANTIZE", "ADD_BIAS", "OUTPUT")
    state = Signal(states.LOAD)
    input_memory = [
        Signal(intbv(0, min=-256, max=256))
        for _ in range(input_size)
    ]
    even_weight_memory = [
        Signal(
            intbv(
                weights[(output_index * 2) * input_size + input_index],
                min=-128,
                max=128,
            )
        )
        for output_index in range(output_size // 2)
        for input_index in range(input_size)
    ]
    odd_weight_memory = [
        Signal(
            intbv(
                weights[(output_index * 2 + 1) * input_size + input_index],
                min=-128,
                max=128,
            )
        )
        for output_index in range(output_size // 2)
        for input_index in range(input_size)
    ]
    matmul_multiplier_memory = [
        Signal(intbv(multiplier, min=-(1 << 31), max=(1 << 31)))
        for multiplier in matmul_multipliers
    ]
    add_bias_offset_memory = [
        Signal(intbv(offset, min=-(1 << 47), max=(1 << 47)))
        for offset in add_bias_offsets
    ]
    input_load_index = Signal(intbv(0, min=0, max=input_size))
    input_index = Signal(intbv(0, min=0, max=input_size))
    output_index = Signal(intbv(0, min=0, max=output_size))
    weight_bank_address = Signal(
        intbv(0, min=0, max=(output_size // 2) * input_size)
    )
    input_value = Signal(intbv(0, min=-256, max=256))
    even_weight_value = Signal(intbv(0, min=-128, max=128))
    odd_weight_value = Signal(intbv(0, min=-128, max=128))
    input_value_extended = Signal(intbv(0, min=-(1 << 15), max=(1 << 15)))
    even_weight_extended = Signal(intbv(0, min=-(1 << 15), max=(1 << 15)))
    odd_weight_extended = Signal(intbv(0, min=-(1 << 15), max=(1 << 15)))
    even_mac_product = Signal(intbv(0, min=-(1 << 15), max=(1 << 15)))
    odd_mac_product = Signal(intbv(0, min=-(1 << 15), max=(1 << 15)))
    even_accumulator = Signal(intbv(0, min=ACCUMULATOR_MIN, max=ACCUMULATOR_MAX))
    odd_accumulator = Signal(intbv(0, min=ACCUMULATOR_MIN, max=ACCUMULATOR_MAX))
    even_matmul_multiplier = Signal(intbv(0, min=-(1 << 31), max=(1 << 31)))
    odd_matmul_multiplier = Signal(intbv(0, min=-(1 << 31), max=(1 << 31)))
    even_add_bias_offset = Signal(intbv(0, min=-(1 << 47), max=(1 << 47)))
    odd_add_bias_offset = Signal(intbv(0, min=-(1 << 47), max=(1 << 47)))
    even_accumulator_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    odd_accumulator_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    even_multiplier_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    odd_multiplier_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    even_matmul_product = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    odd_matmul_product = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    even_matmul_value_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    odd_matmul_value_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    even_add_centered_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    odd_add_centered_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    even_add_multiplier_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    odd_add_multiplier_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    even_add_offset_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    odd_add_offset_extended = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    even_add_product = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    odd_add_product = Signal(intbv(0, min=-(1 << 63), max=(1 << 63)))
    mac_primed = Signal(bool(0))
    output_phase = Signal(bool(0))
    even_matmul_value = Signal(intbv(0)[8:])
    odd_matmul_value = Signal(intbv(0)[8:])
    even_output_quantized = Signal(intbv(0)[8:])
    odd_output_quantized = Signal(intbv(0)[8:])
    even_output_register = Signal(intbv(0)[8:])
    odd_output_register = Signal(intbv(0)[8:])

    @always_comb
    def control_outputs():
        input_ready.next = state == states.LOAD and not reset
        output_valid.next = state == states.OUTPUT and not reset
        output_data.next = odd_output_register if output_phase else even_output_register

    @always(clk.posedge)
    def read_memories():
        input_value.next = input_memory[input_index]

    @always(clk.posedge)
    def read_weight_banks():
        even_weight_value.next = even_weight_memory[weight_bank_address]
        odd_weight_value.next = odd_weight_memory[weight_bank_address]

    @always_comb
    def read_weight():
        weight_bank_address.next = (output_index // 2) * input_size + input_index

    @always_comb
    def select_requantization_constants():
        if output_index % 2 == 0:
            even_matmul_multiplier.next = matmul_multiplier_memory[output_index]
            odd_matmul_multiplier.next = matmul_multiplier_memory[output_index + 1]
            even_add_bias_offset.next = add_bias_offset_memory[output_index]
            odd_add_bias_offset.next = add_bias_offset_memory[output_index + 1]
        else:
            even_matmul_multiplier.next = matmul_multiplier_memory[output_index - 1]
            odd_matmul_multiplier.next = matmul_multiplier_memory[output_index]
            even_add_bias_offset.next = add_bias_offset_memory[output_index - 1]
            odd_add_bias_offset.next = add_bias_offset_memory[output_index]

    @always_comb
    def extend_mac_operands():
        input_value_extended.next = input_value
        even_weight_extended.next = even_weight_value
        odd_weight_extended.next = odd_weight_value

    @always_comb
    def multiply_macs():
        even_mac_product.next = input_value_extended * even_weight_extended
        odd_mac_product.next = input_value_extended * odd_weight_extended

    @always_comb
    def extend_matmul_operands():
        even_accumulator_extended.next = even_accumulator
        odd_accumulator_extended.next = odd_accumulator
        even_multiplier_extended.next = even_matmul_multiplier
        odd_multiplier_extended.next = odd_matmul_multiplier

    @always_comb
    def multiply_matmuls():
        even_matmul_product.next = (
            even_accumulator_extended * even_multiplier_extended
        )
        odd_matmul_product.next = (
            odd_accumulator_extended * odd_multiplier_extended
        )

    @always_comb
    def extend_matmul_values():
        even_matmul_value_extended.next = even_matmul_value
        odd_matmul_value_extended.next = odd_matmul_value

    @always_comb
    def center_matmul_values():
        even_add_centered_extended.next = (
            even_matmul_value_extended - matmul_output_zero_point
        )
        odd_add_centered_extended.next = (
            odd_matmul_value_extended - matmul_output_zero_point
        )

    @always_comb
    def extend_add_operands():
        even_add_multiplier_extended.next = add_multiplier
        odd_add_multiplier_extended.next = add_multiplier
        even_add_offset_extended.next = even_add_bias_offset
        odd_add_offset_extended.next = odd_add_bias_offset

    @always_comb
    def multiply_adds():
        even_add_product.next = (
            even_add_centered_extended * even_add_multiplier_extended
            + even_add_offset_extended
        )
        odd_add_product.next = (
            odd_add_centered_extended * odd_add_multiplier_extended
            + odd_add_offset_extended
        )

    @always_comb
    def requantize_matmuls():
        even_quotient = even_matmul_product >> FIXED_SHIFT
        even_rounded = even_quotient
        if even_matmul_product[FIXED_SHIFT:0] > FIXED_HALF or (
            even_matmul_product[FIXED_SHIFT:0] == FIXED_HALF
            and (even_quotient & 1) != 0
        ):
            even_rounded = even_quotient + 1
        even_quantized = even_rounded + matmul_output_zero_point
        if even_quantized < 0:
            even_matmul_value.next = 0
        elif even_quantized > 255:
            even_matmul_value.next = 255
        else:
            even_matmul_value.next = even_quantized

        odd_quotient = odd_matmul_product >> FIXED_SHIFT
        odd_rounded = odd_quotient
        if odd_matmul_product[FIXED_SHIFT:0] > FIXED_HALF or (
            odd_matmul_product[FIXED_SHIFT:0] == FIXED_HALF
            and (odd_quotient & 1) != 0
        ):
            odd_rounded = odd_quotient + 1
        odd_quantized = odd_rounded + matmul_output_zero_point
        if odd_quantized < 0:
            odd_matmul_value.next = 0
        elif odd_quantized > 255:
            odd_matmul_value.next = 255
        else:
            odd_matmul_value.next = odd_quantized

    @always_comb
    def requantize_adds():
        even_quotient = even_add_product >> FIXED_SHIFT
        even_rounded = even_quotient
        if even_add_product[FIXED_SHIFT:0] > FIXED_HALF or (
            even_add_product[FIXED_SHIFT:0] == FIXED_HALF
            and (even_quotient & 1) != 0
        ):
            even_rounded = even_quotient + 1
        even_quantized = even_rounded + output_zero_point
        if even_quantized < 0:
            even_output_quantized.next = 0
        elif even_quantized > 255:
            even_output_quantized.next = 255
        else:
            even_output_quantized.next = even_quantized

        odd_quotient = odd_add_product >> FIXED_SHIFT
        odd_rounded = odd_quotient
        if odd_add_product[FIXED_SHIFT:0] > FIXED_HALF or (
            odd_add_product[FIXED_SHIFT:0] == FIXED_HALF
            and (odd_quotient & 1) != 0
        ):
            odd_rounded = odd_quotient + 1
        odd_quantized = odd_rounded + output_zero_point
        if odd_quantized < 0:
            odd_output_quantized.next = 0
        elif odd_quantized > 255:
            odd_output_quantized.next = 255
        else:
            odd_output_quantized.next = odd_quantized

    @always(clk.posedge)
    def write_input_memory():
        if not reset and state == states.LOAD and input_valid:
            input_memory[input_load_index].next = input_data

    @always_seq(clk.posedge, reset=reset)
    def sequencer():
        if reset:
            state.next = states.LOAD
            input_load_index.next = 0
            input_index.next = 0
            output_index.next = 0
            even_accumulator.next = 0
            odd_accumulator.next = 0
            mac_primed.next = False
            output_phase.next = False
            even_output_register.next = 0
            odd_output_register.next = 0
        elif state == states.LOAD:
            if input_valid:
                if input_load_index == input_size - 1:
                    input_load_index.next = 0
                    input_index.next = 0
                    output_index.next = 0
                    even_accumulator.next = 0
                    odd_accumulator.next = 0
                    mac_primed.next = False
                    output_phase.next = False
                    state.next = states.MAC
                else:
                    input_load_index.next = input_load_index + 1
        elif state == states.MAC:
            if not mac_primed:
                mac_primed.next = True
                input_index.next = 1
            else:
                even_accumulator.next = even_accumulator + even_mac_product
                odd_accumulator.next = odd_accumulator + odd_mac_product
                if input_index == input_size - 1:
                    state.next = states.MAC_FLUSH
                else:
                    input_index.next = input_index + 1
        elif state == states.MAC_FLUSH:
            even_accumulator.next = even_accumulator + even_mac_product
            odd_accumulator.next = odd_accumulator + odd_mac_product
            mac_primed.next = False
            state.next = states.REQUANTIZE
        elif state == states.REQUANTIZE:
            state.next = states.ADD_BIAS
        elif state == states.ADD_BIAS:
            even_output_register.next = even_output_quantized
            odd_output_register.next = odd_output_quantized
            state.next = states.OUTPUT
        elif state == states.OUTPUT:
            if output_ready:
                if not output_phase:
                    output_phase.next = True
                    output_index.next = output_index + 1
                else:
                    output_phase.next = False
                    if output_index == output_size - 1:
                        input_load_index.next = 0
                        output_index.next = 0
                        state.next = states.LOAD
                    else:
                        output_index.next = output_index + 1
                        input_index.next = 0
                        even_accumulator.next = 0
                        odd_accumulator.next = 0
                        mac_primed.next = False
                        state.next = states.MAC

    return (
        control_outputs,
        read_memories,
        read_weight_banks,
        read_weight,
        select_requantization_constants,
        extend_mac_operands,
        multiply_macs,
        extend_matmul_operands,
        multiply_matmuls,
        extend_matmul_values,
        center_matmul_values,
        extend_add_operands,
        multiply_adds,
        requantize_matmuls,
        requantize_adds,
        write_input_memory,
        sequencer,
    )
