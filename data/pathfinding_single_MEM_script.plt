reset

top_title = "Average Memory usage, pathfinding test, single destination, map Labyrinth Hard"
key_pos_x = 27
key_pos_y = 99

top_title_size = 20

axel_font_size = 16
axel_font_desc_size = 20

key_font_size = 15

set multiplot layout 2,1
set grid
set label 'Memory usage (MB)' at screen 0.01,0.35 rotate by 90 font ", ".axel_font_desc_size

set yrange [0:*]
set xrange [0:410]
set xtics 20 font ", ".axel_font_size
set ytics 10 font ", ".axel_font_size


set key spacing 2
set key font ", ".key_font_size
#set key at key_pos_x, key_pos_y
set key at 35, 75

set title "".top_title font ", ".top_title_size

set tmargin 3
set lmargin 15
set bmargin 2

unset xlabel
set xtics format ""

set ylabel "RAM" offset 0 font ", ".axel_font_desc_size

plot 'IM+PF\pathfinding_MEM_avg_tot.txt' using 1:2 with lines linecolor rgb "blue" linewidth 4 title "IM+PF",\
  'A_+PF\pathfinding_MEM_avg_tot.txt' using 1:2 with lines linecolor rgb "green" linewidth 4 title "A*+PF",\
  'A_\pathfinding_MEM_avg_tot.txt' using 1:2 with lines linecolor rgb "red" linewidth 4 title "A*"

set title ""
set bmargin 4
set tmargin 0
#set ylabel "Memory usage (MB)" offset 0 font ", ".axel_font_desc_size
set xlabel "Units" offset 0,-0.5 font ", ".axel_font_desc_size
set xtics format

set ylabel "VRAM" offset 0 font ", ".axel_font_desc_size

set yrange [0:0.25]
set xrange [0:410]
set xtics 20 font ", ".axel_font_size
set ytics 0.05 font ", ".axel_font_size

set key at 35, 0.246

plot 'IM+PF\pathfinding_MEM_avg_tot.txt' using 1:3 with lines linecolor rgb "blue" linewidth 4 title "IM+PF",\
  'A_+PF\pathfinding_MEM_avg_tot.txt' using 1:3 with lines linecolor rgb "green" linewidth 4 title "A*+PF",\
  'A_\pathfinding_MEM_avg_tot.txt' using 1:3 with lines linecolor rgb "red" linewidth 4 title "A*"

#set ylabel " " offset 0 font ", ".axel_font_desc_size
#set xlabel "Units" offset 0,-0.5 font ", ".axel_font_desc_size
#set xtics format
#set tmargin 0
#set bmargin 4
