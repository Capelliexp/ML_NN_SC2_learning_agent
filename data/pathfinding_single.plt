reset

#top_title = "Average frame time, pathfinding test, single destination, map Labyrinth Hard Two"
top_title = "Path calculation frame, pathfinding test, single destination, map Labyrinth Hard Two"
key_pos_x = 43
#key_pos_y = 138.5
key_pos_y = 3770

top_title_size = 20

axel_font_size = 16
axel_font_desc_size = 20

key_font_size = 20

set yrange [0:*]
set xrange [0:410]
set xtics 20 font ", ".axel_font_size
#set ytics 10 font ", ".axel_font_size
set ytics 200 font ", ".axel_font_size
set grid

set key spacing 2
set key font ", ".key_font_size
set key at key_pos_x, key_pos_y

set ylabel "Frame time (milliseconds)" offset -2 font ", ".axel_font_desc_size
set xlabel "Units" offset 0,-0.5 font ", ".axel_font_desc_size
set title "".top_title font ", ".top_title_size

set bmargin at screen 0.1
set lmargin at screen 0.08

#avg frame time
#plot 'IM+PF\pathfinding_frames_avg.txt' with lines linecolor rgb "blue" linewidth 4 title "IM+PF",\
#  'A_+PF\pathfinding_frames_avg.txt' with lines linecolor rgb "green" linewidth 4 title "A*+PF",\
#  'A_\pathfinding_frames_avg.txt' with lines linecolor rgb "red" linewidth 4 title "A*",\

#start frame time
plot 'IM+PF\pathfinding_frames_start.txt' with lines linecolor rgb "blue" linewidth 4 title "IM+PF",\
  'A_+PF\pathfinding_frames_start.txt' with lines linecolor rgb "green" linewidth 4 title "A*+PF",\
  'A_\pathfinding_frames_start.txt' with lines linecolor rgb "red" linewidth 4 title "A*",\
