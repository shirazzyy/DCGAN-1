# from defectDcgan.avatar_model import defectModel
from defect_model import DefectModel  # wdb 20200325

if __name__ == '__main__':
    defect = DefectModel()
    defect.gen()
    print("图片生成完成.")
