from multiprocessing import Process


def run_parallel(function, args, number_of_cores):
    procs = [Process(target=function, args=[proc_num] + args)
             for proc_num in range(number_of_cores)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
