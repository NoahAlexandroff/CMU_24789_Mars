#!/bin/zsh

start=1000
end=1500
skip=50

for start_ind in {$start..$(($end-$skip))..$skip}
    do
        end_ind=$(($start_ind+$skip))
        python src/scrape_mer_depth_data.py --starting-index $start_ind --ending-index $end_ind &
done
