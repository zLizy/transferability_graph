for CONTAIN_DATASET_FEATURE in True; #False
do
    echo CONTAIN_DATASET_FEATURE-$CONTAIN_DATASET_FEATURE
    for EMBED_DATASET_FEATURE in True; #False
    do
        echo EMBED_DATASET_FEATURE-$EMBED_DATASET_FEATURE
        if [ $CONTAIN_DATASET_FEATURE == False ] && [ $EMBED_DATASET_FEATURE == False ]; then continue; fi 
        for EMBED_MODEL_FEATURE in True; #False
        do
                echo EMBED_MODEL_FEATURE-$EMBED_MODEL_FEATURE
                for CONTAIN_DATA_SIMILARITY in True; #False
                do
                    echo CONTAIN_DATA_SIMILARITY-$CONTAIN_DATA_SIMILARITY
                    for dataset in cifar100 caltech101 dmlab eurosat oxford_iiit_pet oxford_flowers102;
                    # kitti cifar100 resisc45 
                    # kitti cifar100;
                    #diabetic_retinopathy_detection svhn patch_camelyon ;
                    do
                        # kitti cifar100 dmlab clevr diabetic_retinopathy_detection svhn_cropped patch_camelyon
                        # eurosat oxford_iiit_pet oxford_flowers102 resisc45 dtd sun397 caltech101
                        echo dataset-$dataset
                        for complete_model_features in True; # False
                        do
                            echo complete_model_features-${complete_model_features}
                            for FINETUE_RATIOS in 1.0; # 0.4 0.6 0.8
                            do 
                                echo FINETUE_RATIOS-$FINETUE_RATIOS
                                for CONTAIN_MODEL_FEATURE in True; # False True
                                do
                                    echo CONTAIN_MODEL_FEATURE-$CONTAIN_MODEL_FEATURE
                                    if [ $CONTAIN_MODEL_FEATURE == False ] && [ $EMBED_MODEL_FEATURE == False ]; then continue; fi 
                                
                                    for ACCU_THRES in 0.7 0.85 0.9;
                                    do 
                                        echo ACCU_THRES-$ACCU_THRES
                                        for GNN_METHOD in  node2vec; #HGTConv; #GATConv; #'""'; #; #SAGEConv;  #; #;
                                        do
                                            echo GNN_METHOD-$GNN_METHOD
                                            for hidden_channels in 1280; #'""'; #HGTConv; #SAGEConv;  #; #;
                                            do
                                                echo GNN_METHOD-$GNN_METHOD
                                                python3 leave_one_out.py \
                                                        -contain_data_similarity ${CONTAIN_DATA_SIMILARITY} \
                                                        -contain_dataset_feature ${CONTAIN_DATASET_FEATURE} \
                                                        -embed_dataset_feature ${EMBED_DATASET_FEATURE} \
                                                        -contain_model_feature ${CONTAIN_MODEL_FEATURE} \
                                                        -embed_model_feature ${EMBED_MODEL_FEATURE} \
                                                        -complete_model_features ${complete_model_features} \
                                                        -accuracy_thres ${ACCU_THRES} \
                                                        -gnn_method ${GNN_METHOD} \
                                                        -finetune_ratio ${FINETUE_RATIOS} \
                                                        -hidden_channels ${hidden_channels} \
                                                        -test_dataset ${dataset}
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
