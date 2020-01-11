reset

top_title = "Agent Average Score per Iteration"

top_title_size = 40

axel_font_size = 30
axel_font_desc_size = 40

set yrange [-50:370]
set xrange [0:10.25]
set xtics 1 font ", ".axel_font_size scale 200
set ytics 50 font ", ".axel_font_size scale 400
set grid
unset key

set ylabel "Environment Score" offset -8 font ", ".axel_font_desc_size
set xlabel "Agent Training Iteration" offset 0,-1 font ", ".axel_font_desc_size
set title "".top_title font ", ".top_title_size offset 0,1

set bmargin at screen 0.1
set lmargin at screen 0.08

plot 'dqn_mlp_std_simple_data.txt' with lines linecolor rgb "black" linewidth 7
