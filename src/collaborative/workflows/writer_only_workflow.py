class WriterOnlyWorkflow:
    def __init__(self, writer):
        self.writer = writer

    def write(self, content):
        return self.writer.write(content)

    def get_writer(self):
        return self.writer
