for task in reach push pick_and_place
do
    for type in sparse dense
    do
        for p in 1.0 0.75 0.5 0.25 0.0
        do 
            ./spython.sh cldt/utils.py -d datasets/panda_${task}_${type}_1m.pkl -ratio $p
        done
    done
done

