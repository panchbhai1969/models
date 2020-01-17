# REPORTS_DIR=$1
# IMAGES_DIR=$2
DATASET_PATH=$1
LABEL_MAP_PATH=$2
MODEL_NUM=$3
PIPELINE_CONFIG_PATH=$4
NUM_TRAIN_STEPS=$5
USR_NAME=$6
NUM_SHARDS=1
PROJECT_DIR="RATATOUILLE"
DETECTION_MODELS_DIR="${PROJECT_DIR}/DETECTION_MODELS"
USR_DIR="${PROJECT_DIR}/${USR_NAME}"

# REMOVE THE BELOW LINE - ONLY FOR DEVELOPMENT PERIOD
command rm -rf $USR_DIR 

command mkdir $USR_DIR
command mkdir "${USR_DIR}/data" 
command mkdir "${USR_DIR}/models" 

# Intializing log file.
LOG_FILE="${USR_DIR}/status.log"
DUMP_FILE="${USR_DIR}/dump.log"
command touch $LOG_FILE
command touch $DUMP_FILE 

# Redirecting standard output to dump file
{

# exec  | tee $DUMP_FILE
exec 2>&1




MODEL_DIR="${USR_DIR}/models/MODEL"
command mkdir $MODEL_DIR

OUTPUT_DIR="${USR_DIR}/data"

echo "Creating tfRecords" | tee -a $LOG_FILE
command export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
command python3 create_tfRecords.py --dataset_path=$DATASET_PATH --output_dir=$OUTPUT_DIR --label_map_path=$LABEL_MAP_PATH --num_shards=$NUM_SHARDS
echo "tfRecords Created" | tee -a $LOG_FILE

command cp $LABEL_MAP_PATH "${OUTPUT_DIR}/label_map.pbtxt"
echo "Label Map Copied" | tee -a $LOG_FILE

echo "Importing Model" | tee -a $LOG_FILE
if [ $MODEL_NUM == '1' ]
then
    command cp -r "${DETECTION_MODELS_DIR}/faster_rcnn_resnet50_coco_2018_01_28/"*  "${MODEL_DIR}/"
fi
TRAIN_DIR="${MODEL_DIR}/train"
command mkdir $TRAIN_DIR
echo "Model Imported Successfully." | tee -a $LOG_FILE

echo "Copying Config File" | tee -a $LOG_FILE
command cp $PIPELINE_CONFIG_PATH "${MODEL_DIR}/pipeline.config"
PIPELINE_CONFIG_PATH="${MODEL_DIR}/pipeline.config"
command sed -i "s/bhadwa/${USR_NAME}/g" "${PIPELINE_CONFIG_PATH}"
echo "Config File Copied" | tee -a $LOG_FILE

exit 0

echo "Training started." | tee -a $LOG_FILE
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
command python3 object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${TRAIN_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
echo "Training ended." | tee -a $LOG_FILE

echo "Exporting Inference Graph" | tee -a $LOG_FILE
INFERENCE_GRAPH_PATH="${TRAIN_DIR}/inference_graph"
command mkdir $INFERENCE_GRAPH_PATH
command python3 object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path $PIPELINE_CONFIG_PATH \
    --trained_checkpoint_prefix "${TRAIN_DIR}/model.ckpt-${NUM_TRAIN_STEPS}" path/to/model.ckpt \
    --output_directory $INFERENCE_GRAPH_PATH
echo "Inference Graph Exported." | tee -a $LOG_FILE




# Redirecting outputs to default
exec 1>&-   #closes FD 1 (logfile)
exec 2>&-   #closes FD 2 (logfile)
exec 2>&4   #restore stderr
exec 1>&3   #restore stdout


} | tee -a $DUMP_FILE