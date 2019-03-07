from vvc import config, vvc

if __name__ == '__main__':
    for model in config.models:
        vvc.VVC('MOV_0861.mp4', model).count()
    