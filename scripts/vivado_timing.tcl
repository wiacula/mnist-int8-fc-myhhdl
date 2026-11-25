if {[llength $argv] != 4} {
    error "Usage: vivado -mode batch -source scripts/vivado_timing.tcl -tclargs <rtl_file> <timing_report> <top_name> <clock_period_ns>"
}

set rtl_file [file normalize [lindex $argv 0]]
set timing_report [file normalize [lindex $argv 1]]
set top_name [lindex $argv 2]
set clock_period [lindex $argv 3]

if {![string is double -strict $clock_period] || $clock_period <= 0.0} {
    error "Clock period must be a positive number in nanoseconds"
}

set report_root [file rootname $timing_report]
set path_report "${report_root}_paths.rpt"
set utilization_report "${report_root}_utilization.rpt"

file mkdir [file dirname $timing_report]
cd [file dirname $rtl_file]
read_verilog $rtl_file
synth_design -top $top_name -part xc7z020clg400-1
create_clock -name core_clk -period $clock_period [get_ports clk]
opt_design
place_design
phys_opt_design
route_design

report_timing_summary -delay_type min_max -report_unconstrained -file $timing_report
report_timing -delay_type max -max_paths 10 -file $path_report
report_utilization -file $utilization_report

puts "Timing summary: $timing_report"
puts "Critical paths: $path_report"
puts "Utilization: $utilization_report"
