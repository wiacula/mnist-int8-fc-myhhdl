set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set project_dir [file join $repo_root vivado_project]
set project_name mnist_fc
set project_file [file join $project_dir "${project_name}.xpr"]
set source_dir [file join $project_dir src]
set rtl_file [file join $source_dir mnist_fc_serial.v]
set memory_files [list \
    [file join $source_dir even_weight_memory.mem] \
    [file join $source_dir odd_weight_memory.mem]]

foreach source [concat [list $rtl_file] $memory_files] {
    if {![file isfile $source]} {
        error "Required project source is missing: $source"
    }
}

if {[file isfile $project_file]} {
    open_project $project_file
} else {
    create_project $project_name $project_dir -part xc7z020clg400-1
}

set_property part xc7z020clg400-1 [current_project]
set_property target_language Verilog [current_project]
set source_fileset [get_filesets sources_1]
set_property top mnist_fc_serial $source_fileset

foreach source [concat [list $rtl_file] $memory_files] {
    if {[llength [get_files -quiet $source]] == 0} {
        add_files -norecurse -fileset sources_1 $source
    }
}

foreach memory_file $memory_files {
    set_property file_type {Memory Initialization Files} \
        [get_files -quiet $memory_file]
}

update_compile_order -fileset sources_1
close_project
puts "Vivado project created or updated: $project_file"
