# -*- coding:utf-8 -*-
import json
import os

from bert_cause.base.server.tcp_client import TcpClient
from bert_cause.base.helper.json_helper import parse_json, center_json
from bert_cause.base.helper.args_helper import get_client_args


class Predictor:

    def __init__(self, ip='localhost', port=5555, port_out=5556, timeout=-1, use_center=True):
        """
            初始化模型、配置
        """
        self.client = TcpClient(ip=ip, port=port, port_out=port_out, exceed_time=timeout)
        self.use_center = use_center
        pass

    def predict(self, content: dict) -> dict:
        """
        输入标注格式，已转为dict
        输出同标注格式，dict格式
        :param content:
        :return str:
        """
        text, centers = parse_json(content)
        res = self.client.request([text])
        if self.use_center:  # 是否按照中心词匹配
            res = center_json(res[0], centers)
        return res


if __name__ == "__main__":
    args = get_client_args()

    input_file = os.path.join(args.root_path, args.input_file)
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    predictor = Predictor(args.host, args.port, args.port_out, args.timeout, args.use_center)
    outputs = []

    for line in lines:
        output = predictor.predict(json.loads(line))
        outputs.append(output)
        print("output:", output)

    if args.result_to_file:
        output_file = os.path.join(args.root_path, args.output_file)
        with open(output_file, 'w', encoding='utf-8') as file:
            for output_line in outputs:
                file.write(str(output_line) + '\n')
