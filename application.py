import tornado.web
from tornado.ioloop import IOLoop
from src.handlers.predict import ModelHandler


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('Hello, world')


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/?", MainHandler),
            (r"/predict/?", ModelHandler),
            
        ]
        tornado.web.Application.__init__(self, handlers)


def main():
    app = Application()
    app.listen(8002)
    print("App Starting")
    IOLoop.instance().start()

if __name__ == '__main__':
    main()
