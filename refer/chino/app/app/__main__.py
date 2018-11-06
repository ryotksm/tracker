import sys
sys.path.insert(0, '.')


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')

    from app import server
    server.start()
