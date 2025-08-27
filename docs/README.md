# 先切到docs目录下
cd docs

# 从./source/conf.py中读取文档设置，并调用从写好的rst或md的原文文档中在./build/gettext生成所有文档文件对应的.pot文件
sphinx-build -b gettext ./source build/gettext

# 在docs目录下
sphinx-intl update -p ./build/gettext -l zh_CN

# 编译 gpt-po
cd ~/coding/gpt-po/gpt-po-master
tsc

# gpt-po 翻译
node ./src/index.js --dir /home/yanghao/coding/nccl-2.19-docs-cn/docs/locales/zh_CN/LC_MESSAG

# 构建英文版
make html

# 构建中文版
sphinx-build -b html -D language=zh_CN ./source/ build/html/zh_CN

# 上传至github后，read the docs自动更新
https://nccl-219-docs-cn.readthedocs.io/en/latest/