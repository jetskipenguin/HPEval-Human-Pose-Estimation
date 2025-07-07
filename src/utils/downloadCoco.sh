cd ..
cd ..
if [ -d "COCO" ]; then
  echo "COCO directory already exists!"
  exit 1
fi
mkdir COCO
cd COCO
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
