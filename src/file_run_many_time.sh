for num_depth in 128; do
    for step in 0.01 0.1 0.5 0.9; do
        for alpha in 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5; do
            python3 run_grand_ex.py --depth $num_depth --discritize_type norm --step_size $step --trunc_alpha $alpha --dataset Cora
            #python3 run_grand_ex.py --depth $num_depth --discritize_type norm --step_size $step --trunc_alpha $alpha --dataset Citeseer
        done
    done
done
