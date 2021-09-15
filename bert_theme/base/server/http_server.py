# -*- coding: UTF-8 -*-
import codecs
import json
import os
from multiprocessing import Process

from flask import Flask, request
from flask_compress import Compress
from flask_cors import CORS
from flask_json import FlaskJSON, as_json, JsonError

from bert_cause.base.helper.log_helper import get_logger
from bert_theme.base.server.tcp_client import TcpClient

logger = get_logger()


class HttpProxy(Process):
    def __init__(self, args):
        self.args = args
        Process.__init__(self)

    def create_flask_app(self):
        # 启动 tcp 客户
        client = TcpClient(
            ip=self.args.host,
            port=self.args.port,
            port_out=self.args.port_out,
            exceed_time=self.args.timeout)

        app = Flask(__name__)

        @app.route('/theme', methods=['GET'])
        @as_json
        def theme_query():
            with codecs.open(os.path.join(self.args.output_dir, 'category3d.json'), 'r', encoding='utf-8') as file:
                line = file.readline()
                try:
                    context_dict = json.loads(line)  # 将 json 格式的字符串转化为字典
                except json.decoder.JSONDecodeError as e:
                    print(e)
            return context_dict

        @app.route('/encode/theme', methods=['POST'])
        @as_json
        def encode_theme_query():
            data = request.form if request.form else request.json
            try:
                res = client.request(data['theme'])
                new_res = []
                for json_line in res:
                    json_line = json.dumps(json_line, ensure_ascii=False)
                    new_res.append(json_line)
                logger.info("[ENCODER REQUEST] Predict Result: %s " % new_res)
                return {'predict': new_res}
            except Exception as e:
                logger.error('error when handling HTTP request', exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        CORS(app, origins=self.args.cors)  # Access-Control-Allow-Origin
        FlaskJSON(app)
        Compress().init_app(app)
        return app

    def run(self):
        app = self.create_flask_app()
        app.run(port=self.args.http_port, host=self.args.host, threaded=True)


def run_http_server(args):
    """ 开启 http 服务 """
    proc_proxy = HttpProxy(args)
    proc_proxy.start()
    return proc_proxy
