Extract 1 minute

ffmpeg -i Ch2_20181110121206.mp4 -ss 0 -t 60 Ch2_20181110121206_1min.mp4

Extract a frame by minute

ffmpeg -i Ch2_20181110121206.mp4 -r 1/60 "$filename%03d.jpg"
