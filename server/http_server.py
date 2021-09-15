# -*- coding: UTF-8 -*-
import codecs
import json
from multiprocessing import Process

from flask import Flask, request
from flask_compress import Compress
from flask_cors import CORS
from flask_json import FlaskJSON, as_json, JsonError

from bert_cause.base.helper.json_helper import parse_json, load_jsons
from bert_cause.base.helper.log_helper import get_logger
from bert_cause.base.server.tcp_client import TcpClient as CauseTcpClient
from bert_theme.base.server.tcp_client import TcpClient as ThemeTcpClient

logger = get_logger()


class HttpProxy(Process):
    def __init__(self, args):
        self.args = args
        Process.__init__(self)

    def create_flask_app(self):

        theme_client = ThemeTcpClient(
            ip=self.args.host,
            port=self.args.theme_port,
            port_out=self.args.theme_port_out,
            exceed_time=self.args.timeout)

        cause_client = CauseTcpClient(
            ip=self.args.host,
            port=self.args.cause_port,
            port_out=self.args.cause_port_out,
            exceed_time=self.args.timeout)

        app = Flask(__name__)

        @app.route('/encode/text', methods=['POST'])
        @as_json
        def encode_text_query():
            data = request.form if request.form else request.json
            try:
                texts = data['text']
                if type(texts) is not list:
                    texts = [texts]
                res = cause_client.request(texts)
                new_res = []
                for json_line in res:
                    json_line = json.dumps(json_line, ensure_ascii=False)
                    new_res.append(json_line)
                logger.info("[ENCODER REQUEST] Predict Result: %s " % new_res)
                return {'predict': new_res}
            except Exception as e:
                logger.error('error when handling HTTP request', exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        @app.route('/encode/json', methods=['POST'])
        @as_json
        def encode_json_query():
            data = request.form if request.form else request.json
            try:
                text, _ = parse_json(data['json'])
                if text is None:
                    return {'predict': [], 'msg': 'Your json format is incorrect!'}
                res = cause_client.request([text])
                new_res = []
                for json_line in res:
                    json_line = json.dumps(json_line, ensure_ascii=False)
                    new_res.append(json_line)
                logger.info("[ENCODER REQUEST] Predict Result: %s " % new_res)
                return {'predict': new_res}
            except Exception as e:
                logger.error('error when handling HTTP request', exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        @app.route('/encode/theme', methods=['POST'])
        @as_json
        def encode_theme_query():
            data = request.form if request.form else request.json
            try:
                texts = data['theme']
                if type(texts) is not list:
                    texts = [texts]
                texts = load_jsons(texts)
                res = theme_client.request(texts)  # [dict]
                new_res = []
                for json_dict in res:
                    assert type(json_dict) is dict
                    json_line = json.dumps(json_dict, ensure_ascii=False)
                    new_res.append(json_line)
                logger.info("[ENCODER REQUEST] Predict Result: %s " % new_res)
                return {'predict': new_res}
            except Exception as e:
                logger.error('error when handling HTTP request', exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        @app.route('/theme', methods=['GET'])
        @as_json
        def theme_query():
            with codecs.open('category3d.json', 'r', encoding='utf-8') as file:
                line = file.readline()
                try:
                    context_dict = json.loads(line)  # 将 json 格式的字符串转化为字典
                except json.decoder.JSONDecodeError as e:
                    print(e)
            return context_dict

        CORS(app, origins=self.args.cors)
        FlaskJSON(app)
        Compress().init_app(app)
        return app

    def run(self):
        app = self.create_flask_app()
        app.run(port=self.args.http_port, host=self.args.host, threaded=True)
