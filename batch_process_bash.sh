source ~/.bashrc
conda activate /users/rfu7/data/anaconda/object_env
cd object_estimation

ITH=$1
SESSION=$2
SCANPATH=$3

echo "########################## TRACKING OBJECT POSE ################################"
python scripts/1_object_tracking.py --ith $ITH --session $SESSION --scan_path $SCANPATH --save_seg_frame