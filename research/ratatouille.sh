REPORTS_DIR=$1
IMAGES_DIR=$2
LABEL_MAP_PATH=$3
MODEL_NUM=$4
PIPELINE_CONFIG_PATH=$5
NUM_TRAIN_STEPS=$6
NUM_SHARDS=1
USR_NAME="bhadwa"
PROJECT_DIR="RATATOUILLE"
DETECTION_MODELS_DIR="${PROJECT_DIR}/DETECTION_MODELS"
USR_DIR="${PROJECT_DIR}/${USR_NAME}"

# REMOVE THE BELOW LINE - ONLY FOR DEVELOPMENT PERIOD
command rm -rf $USR_DIR 

command mkdir $USR_DIR
command mkdir "${USR_DIR}/data" 
command mkdir "${USR_DIR}/models" 

MODEL_DIR="${USR_DIR}/models/MODEL"
command mkdir $MODEL_DIR

OUTPUT_DIR="${USR_DIR}/data"

echo "Creating tfRecords"
command export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
command python3 create_tfRecords.py --reports_dir=$REPORTS_DIR --images_dir=$IMAGES_DIR --output_dir=$OUTPUT_DIR --label_map_path=$LABEL_MAP_PATH --num_shards=$NUM_SHARDS
echo "tfRecords Created"

command cp $LABEL_MAP_PATH "${OUTPUT_DIR}/label_map.pbtxt"
echo "Label Map Copied"

echo "Importing Model"
if [ $MODEL_NUM == 1 ]
then
    command cp -r "${DETECTION_MODELS_DIR}/faster_rcnn_resnet50_coco_2018_01_28/"*  "${MODEL_DIR}/"
fi
TRAIN_DIR="${MODEL_DIR}/train"
command mkdir $TRAIN_DIR
echo "Model Imported Successfully."

echo "Copying Config File"
command cp $PIPELINE_CONFIG_PATH "${MODEL_DIR}/pipeline.config"
PIPELINE_CONFIG_PATH="${MODEL_DIR}/pipeline.config"
echo "Config File Copied"

echo "Training started."
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
command python3 object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${TRAIN_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
echo "Training ended."

echo "Exporting Inference Graph"
INFERENCE_GRAPH_PATH="${TRAIN_DIR}/inference_graph"
command mkdir $INFERENCE_GRAPH_PATH
command python3 object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path $PIPELINE_CONFIG_PATH \
    --trained_checkpoint_prefix "${TRAIN_DIR}/model.ckpt-${NUM_TRAIN_STEPS}" path/to/model.ckpt \
    --output_directory $INFERENCE_GRAPH_PATH