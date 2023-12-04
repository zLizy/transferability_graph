echo 
echo ====== new command ========
for CONTAIN_DATASET_FEATURE in True; #  True  False
do
    echo CONTAIN_DATASET_FEATURE-$CONTAIN_DATASET_FEATURE
    for CONTAIN_DATA_SIMILARITY in True; #True
    do
        echo CONTAIN_DATA_SIMILARITY-$CONTAIN_DATA_SIMILARITY
        for dataset in svhn stanfordcars dtd caltech101 oxford_iiit_pet oxford_flowers102
        do
            #  smallnorb_label_azimuth eurosat diabetic_retinopathy_detection 
            #   kitti smallnorb_label_elevation cifar100
            echo dataset-$dataset
            for complete_model_features in True; # False
            do
                echo complete_model_features-${complete_model_features}
                for top_pos_K in 0.5; #75; #100 150 200; #50
                do
                    echo top_pos_K-$top_pos_K
                    for top_neg_K in 0.5; #20 ; #30 40
                    do
                        echo top_neg_K-$top_neg_K
                        for accu_neg_thres in 0.5; # 0.2 0.3 0.4; # 0.4 0.6 0.8
                        do 
                            echo accu_neg_thres-$accu_neg_thres
                            for dataset_embed_method in domain_similarity; #; #task2vec
                            do
                                for CONTAIN_MODEL_FEATURE in False; # False True
                                do
                                    echo CONTAIN_MODEL_FEATURE-$CONTAIN_MODEL_FEATURE
                                    for accu_pos_thres in 0.5; #0.6 0.7 -1; #0.85 0.75 
                                    do 
                                        echo accu_pos_thres-$accu_pos_thres
                                        for hidden_channels in 128; #64 32
                                        do
                                            echo distance-$distance
                                            for GNN_METHOD in lr_node2vec+ lr_homo_SAGEConv lr_homoGATConv
                                                #  lr_homo_SAGEConv lr_homoGATConv lr_homoGCNConv lr_homoGATConv lr_node2vec lr_node2vec+_without_transfer lr_node2vec_without_transfer
                                                #                   node2vec+_w2v rf_node2vec+ lr_node2vec+_all lr_node2vec_all
                                                #                   homo_SAGEConv_e2e homo_GATConv_e2e  homo_GCNConv_e2e
                                                #                   rf_node2vec_without_accuracy rf_node2vec+_without_accuracy
                                                #                   lr_node2vec_without_accuracy lr_node2vec+_without_accuracy \
                                                #                   lr_homo_GATConv_without_accuracy \ homo_SAGEConv_trained_on_transfer
                                                #                   lr_homo_SAGEConv_without_accuracy;
                                                #                   \    lr_homo_SAGEConv_e2e lr_homo_GATConv_e2e lr_homo_GCNConv_e2e
                                                #                   GATConv GATConv_trained_on_transfer GATConv_without_transfer \
                                                #                   node2vec SAGEConv  GATConv_without_transfer HeteroGNN  HGTConv  GATConv node2vec_without_transfer HeteroGNN SAGEConv node2vec; # #SAGEConv GATConv HGTConv; 
                                            do
                                                echo GNN_METHOD-$GNN_METHOD
                                                for finetune_ratio in 1.0; #1.0; #; # 0.3 0.7 0.5 
                                                do
                                                    echo finetune_ratio-$finetune_ratio
                                                    python3 run.py \
                                                            -contain_data_similarity ${CONTAIN_DATA_SIMILARITY} \
                                                            -contain_dataset_feature ${CONTAIN_DATASET_FEATURE} \
                                                            -contain_model_feature ${CONTAIN_MODEL_FEATURE} \
                                                            -complete_model_features ${complete_model_features} \
                                                            -gnn_method ${GNN_METHOD} \
                                                            -test_dataset ${dataset} \
                                                            -top_neg_K ${top_neg_K} \
                                                            -top_pos_K ${top_pos_K} \
                                                            -accu_neg_thres ${accu_neg_thres} \
                                                            -accu_pos_thres ${accu_pos_thres} \
                                                            -hidden_channels ${hidden_channels} \
                                                            -finetune_ratio ${finetune_ratio} \
                                                            -dataset_embed_method ${dataset_embed_method}                                                                                   ## -embed_dataset_feature ${EMBED_DATASET_FEATURE} \
                                                    ## -embed_model_feature ${EMBED_MODEL_FEATURE} \
                                                    ## -accuracy_thres ${ACCU_THRES} \
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
done
