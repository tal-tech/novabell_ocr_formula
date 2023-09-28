# NovaBell_OCR_ FORMULA

OCR公式识别
### Python版本
python 3.5

### Installation
从FTP下载文本和检测模型，放到./models中
```shell
pip instsall -r requirements.txt
```
### Interface
interface.py
```python
import interface
text_interpreter = interface.TextInterpreter()
image = # load an image with openCV
final_result,interpretor_timer = text_interpreter.interpret(image)
```

### Output
final_result - 图片的文本识别结果
```json
{
    'text':[
        {
            'location': [76,436,178,483], 
            'content': '第三点'
        },
        ...
    ],
    'char_count': 219
}
```
interpretor_timer - 接口运行计时器
```json
{
    'preprocess': 0.0017902851104736328,     # 前处理时间
    'decode': 0.00014472007751464844,        # 识别编码时间
    'det_model': 0.3733797073364258,         # 检测内部模型时间
    'recognizor_cost': 0.32707762718200684,  # 识别消耗时间
    'batches': 1,                            # 运行批次
    'detector_cost': 0.3741745948791504,     # 检测消耗时间
    'total_cost': 0.7013118267059326,        # 总耗时
    'num_lines': 21,                         # 文本行数
    'rec_model': 0.3250746726989746,         # 识别内部模型时间
    'cut_images': 2.4557113647460938e-05,    # 切分图片时间
    'decorate_cost': 5.9604644775390625e-05  # 编码结果时间
}
```
### Demo
```shell
# run test scripts.
python tools/test.py
```