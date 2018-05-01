# Style_Transfer
This projects tries to implement style transfer.
https://arxiv.org/pdf/1705.08086.pdf

Download weights from the given link and place it in a folder.
https://drive.google.com/file/d/0B8_MZ8a8aoSeWm9HSTdXNE9Eejg/view


To execute the files

Example commands
----------------
python style_transfer.py --alpha 0.9 --style-path 2.jpg --content-path 21.jpg --out-path /home/goutham/test --live-path 0 --weight-path /home/goutham/WCT-TF-master/models2/models

python style_mask.py --style-path /home/goutham/WCT-TF-master/samples  --content-path 21.jpg --out-path /home/goutham/test --weight-path /home/goutham/WCT-TF-master/models2/models

python live_demo.py --style-path /home/goutham/WCT-TF-master/samples/style_1.png --out-path /home/goutham/test --weight-path /home/goutham/WCT-TF-master/models2/models

Generalized commands
--------------------
python style_transfer.py --alpha 0.9 --style-path *style_img_path* --content-path *content_img_path* --out-path *out_img_folder_path* --live-path 0 --weight-path *weight_path*

python style_mask.py --style-path *style_imgs_folder_path* --content-path *content_img_path* --out-path *out_img_folder_path* --weight-path /*weight_path*
