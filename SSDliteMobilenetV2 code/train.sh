export PYTHONPATH=$PYTHONPATH:/nfs/private/tfmodels/research:/nfs/private/tfmodels/research/slim
PIPELINE_CONFIG_PATH=/nfs/private/tfmodels/research/My_object_detection_SSDlite_mobilenet_v2/models/ssdlite_mobilenet_v2_coco.config
MODEL_DIR=/nfs/private/tfmodels/research/My_object_detection_SSDlite_mobilenet_v2/models/model/train
python /nfs/private/tfmodels/research/object_detection/model_main.py \
    --logtostderr \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
