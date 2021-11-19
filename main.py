import prepare_data

prepare = prepare_data.PrepareData("/media/Shared Disk/Dataset/Imagens-Labor")
prepare.separe_images_size((1270, 1012), dst1="/media/Shared Disk/Dataset/high/",
                           dst2="/media/Shared Disk/Dataset/mid/")