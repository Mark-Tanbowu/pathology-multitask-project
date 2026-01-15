注意力模块
========

文件
----
- se_block.py：Squeeze-and-Excitation 通道注意力。
- cbam_block.py：CBAM 通道+空间注意力。

改进建议
--------
- 将注入点（encoder/skip/decoder）配置化，便于消融。
- 补充前向形状/参数量单测与性能对比记录。
