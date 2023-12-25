for CONTAIN_DATASET_FEATURE in True False; #  True  False
do
    echo CONTAIN_DATASET_FEATURE-$CONTAIN_DATASET_FEATURE
    
    for CONTAIN_DATA_SIMILARITY in True; #True
    do
        echo CONTAIN_DATA_SIMILARITY-$CONTAIN_DATA_SIMILARITY
        for dataset in stanfordcars; #cifar100 kitti; # svhn_cropped oxford_flowers102 diabetic_retinopathy_detection dmlab dtd resisc45 patch_camelyon sun397 caltech101 eurosat oxford_iiit_pet;
        do
            # kitti cifar100 dmlab clevr diabetic_retinopathy_detection svhn_cropped patch_camelyon
            # eurosat oxford_iiit_pet oxford_flowers102 resisc45 dtd sun397 caltech101
            echo dataset-$dataset
            for complete_model_features in True; # False
            do
                echo complete_model_features-${complete_model_features}
                for top_pos_K in 0.5; #75; #100 150 200; #50
                do
                    echo top_pos_K-$top_pos_K
                    #if [ $CONTAIN_DATASET_FEATURE == False ] && [ $EMBED_DATASET_FEATURE == False ]; then continue; fi 
                    for top_neg_K in 0.2; #20 ; #30 40
                    do
                            echo top_neg_K-$top_neg_K
                            for accu_neg_thres in 0.3; # 0.2 0.3 0.4; # 0.4 0.6 0.8
                            do 
                                echo accu_neg_thres-$accu_neg_thres
                                for CONTAIN_MODEL_FEATURE in False; # False True
                                do
                                    echo CONTAIN_MODEL_FEATURE-$CONTAIN_MODEL_FEATURE
                                    #if [ $CONTAIN_MODEL_FEATURE == False ] && [ $EMBED_MODEL_FEATURE == False ]; then continue; fi 
                                    for accu_pos_thres in 0.6; #0.6 0.7 -1; #0.85 0.75 
                                    do 
                                        echo accu_pos_thres-$accu_pos_thres
                                        for distance_thres in -1;
                                        do
                                            echo distance-$distance
                                            for GNN_METHOD in node2vec node2vec+ \
                                                              SAGEConv SAGEConv_trained_on_transfer SAGEConv_without_transfer \
                                                              homo_SAGEConv \
                                                              GATConv GATConv_trained_on_transfer GATConv_without_transfer \
                                                              homo_GATConv; # node2vec SAGEConv  GATConv_without_transfer HeteroGNN  HGTConv  GATConv node2vec_without_transfer HeteroGNN SAGEConv node2vec; # #SAGEConv GATConv HGTConv; 
                                            do
                                                echo GNN_METHOD-$GNN_METHOD
                                                for hidden_channels in 128; # 1280
                                                do
                                                    echo GNN_METHOD-$GNN_METHOD
                                                    python3 run.py \
                                                            -contain_data_similarity ${CONTAIN_DATA_SIMILARITY} \
                                                            -contain_dataset_feature ${CONTAIN_DATASET_FEATURE} \
                                                            -contain_model_feature ${CONTAIN_MODEL_FEATURE} \
                                                            -complete_model_features ${complete_model_features} \
                                                            -gnn_method ${GNN_METHOD} \
                                                            -hidden_channels ${hidden_channels} \
                                                            -test_dataset ${dataset} \
                                                            -top_neg_K ${top_neg_K} \
                                                            -top_pos_K ${top_pos_K} \
                                                            -accu_neg_thres ${accu_neg_thres} \
                                                            -accu_pos_thres ${accu_pos_thres} \
                                                            -distance_thres ${distance_thres}
                                                    ## -embed_dataset_feature ${EMBED_DATASET_FEATURE} \
                                                    ## -embed_model_feature ${EMBED_MODEL_FEATURE} \
                                                    ## -accuracy_thres ${ACCU_THRES} \
                                                    ## -finetune_ratio ${FINETUE_RATIOS} \
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            
        done
    done
done
