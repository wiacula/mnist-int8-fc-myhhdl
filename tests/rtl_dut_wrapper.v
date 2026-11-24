module tb_mnist_fc_serial;
    reg clk;
    reg reset;
    reg [7:0] input_data;
    reg input_valid;
    wire input_ready;
    wire [7:0] output_data;
    wire output_valid;
    reg output_ready;

    initial begin
        $from_myhdl(clk, reset, input_data, input_valid, output_ready);
        $to_myhdl(input_ready, output_data, output_valid);
    end

    mnist_fc_serial dut (
        .clk(clk),
        .reset(reset),
        .input_data(input_data),
        .input_valid(input_valid),
        .input_ready(input_ready),
        .output_data(output_data),
        .output_valid(output_valid),
        .output_ready(output_ready)
    );
endmodule
