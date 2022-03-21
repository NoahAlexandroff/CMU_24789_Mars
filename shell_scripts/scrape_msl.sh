#!/bin/zsh

start=0
end=18500
skip=500

for start_ind in {$start..$(($end-$skip))..$skip}
    do
        end_ind=$(($start_ind+$skip))
        python src/scrape_msl_depth_data.py --starting-index $start_ind --ending-index $end_ind &
done
