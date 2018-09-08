import os
import cv2
import selectivesearch
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


# 搜索数字区域
def find_num_regions(img):
    # img_not = 255 - img
    img_lbl, regions = selectivesearch.selective_search(img, scale=500
                                                        , sigma=0.9
                                                        , min_size=200)
    # 重复、面积筛选
    candidates = []
    for r in regions:
        if r['rect'] in candidates:
            continue
        if r['size'] < 200 or r['size'] > 20000:
            continue
        x, y, w, h = r['rect']
        # if w / h > 1.2 or h / w > 1.2 :
        #     continue
        # candidates.append((x , y ,w , h))
        candidates.append((x, y, w, h))
    # print(len(candidates))
    # 取最大圈，消除小圈
    num_array = []
    for i in candidates:
        if len(num_array) == 0:
            num_array.append(i)
        else:
            content = False
            replace = -1
            for index, j in enumerate(num_array):
                if i[0] >= j[0] and i[0] + i[2] <= j[0] + j[2] \
                        and i[1] >= j[1] and i[1] + i[3] <= j[1] + j[3]:
                    content = True
                    break
                elif i[0] <= j[0] and i[0] + i[2] >= j[0] + j[2] \
                        and i[1] <= j[1] and i[1] + i[3] >= j[1] + j[3]:
                    replace = index
                    break
            if not content:
                if replace >= 0:
                    num_array[replace] = i
                else:
                    num_array.append(i)
    return num_array

# 根据区域判断数字串
def search_numeric_strings(img):
    numeric_strings_regions = []
    regions = find_num_regions(img)
    # 根据 x 排序
    ordered_regions = sorted(regions ,  key = lambda region : region[0])

    while ordered_regions:
        numericstring_regions = []
        first_region = ordered_regions.pop(0)
        numericstring_regions.append(first_region)
        index = 0
        # 遍历所有regions，弹出符合条件的region
        while index < len(ordered_regions):
            if abs(ordered_regions[index][1] - first_region[1]) < first_region[3] :
                numericstring_regions.append(ordered_regions.pop(index))
            else :
                index += 1
        numeric_strings_regions.append(numericstring_regions)

    return numeric_strings_regions



# 实现单独验证该模块功能
if __name__ == '__main__':
    pwd = os.getcwd()
    project_path = os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
    img_path = os.path.join( project_path,'./test_pic/13.jpg' )

    img = cv2.imread(img_path)

    numeric_strings_regions = search_numeric_strings(img)
    fig_helper , ax_helpers = plt.subplots(ncols = 1 , nrows = len(numeric_strings_regions)  )

    # 可视化标定为同一数字串的regions
    for index , numericstring_regions in enumerate(numeric_strings_regions):
        ax_helpers[index].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        for region in numericstring_regions :
            x , y , w , h  = (region[0] , region[1] , region[2] , region[3] )
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor= 'red', linewidth=1)
            ax_helpers[index].add_patch(rect)

    plt.show()





