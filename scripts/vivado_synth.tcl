if {[llength $argv] != 2} {
    error "Usage: vivado -mode batch -source scripts/vivado_synth.tcl -tclargs <rtl_file> <report_file>"
}

set rtl_file [lindex $argv 0]
set report_file [lindex $argv 1]

set rtl_file [file normalize $rtl_file]
set report_file [file normalize $report_file]
cd [file dirname $rtl_file]
read_verilog $rtl_file
synth_design -top mnist_fc_serial -part xc7z020clg400-1
report_utilization -file $report_file

set bram_cells [get_cells -hierarchical -filter {REF_NAME =~ RAMB*}]
puts "Inferred BRAM cells: $bram_cells"
