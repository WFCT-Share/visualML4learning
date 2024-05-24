import time
import json
import os
import pandas as pd

"""zh：当前包仅用于开发使用，功能尚不完善。所有的日志功能均通过修饰器进行实现"""


def logger_ini():
    """初始化日志文件配置文件，使用日志模块前请先初始化，只需要初始化一次否则无法正常使用"""
    log_front = r".\log\\"
    directory_list = [r".\log", log_front + r"jsons\\",
                      log_front + r"txts\\",
                      log_front + r"csvs\\",
                      log_front + r"excels\\"]
    for directory in directory_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d--%H-%M-%S", current_time)
    log_name = "log_" + formatted_time
    with open(r'log\log_config.json', mode='w', encoding='utf-8') as config:
        log_path_data = {"json_log_path": directory_list[1] + log_name + ".json",
                         "txt_log_path": directory_list[2] + log_name + ".txt",
                         "csv_log_path": directory_list[3] + log_name + ".csv",
                         "excel_log_path": directory_list[4] + log_name + ".xlsx"}
        json.dump(log_path_data, config, ensure_ascii=False, indent=4)

    with open(log_path_data['json_log_path'], mode='w', encoding='utf-8') as logging:
        ini_data = ["程序开始执行"]
        json.dump(ini_data, logging, ensure_ascii=False, indent=4)
    with open(log_path_data['txt_log_path'], mode='w', encoding='utf-8') as logging:
        ini_data = "程序开始执行"
        logging.write(ini_data)
    columns = ['调用模块', '调用函数', '运行时间', '返回值', '备注']
    df = pd.DataFrame(columns=columns)
    df.to_csv(log_path_data["csv_log_path"], index=False)
    df.to_excel(log_path_data["excel_log_path"], index=False)


def log4json(module,
             remarks=None):
    """
             一个用于记录日志的装饰器，接受日志目录和日志编写模式的参数输入，在日志中记录运行的函数和运行的时间，仅适用于返回单个值的函数

             Args:
                 module(str):在面向对象编程中调用的模块的名称，方便开发进行调试时明确模块
                 remarks(str):备注，便于开发者在调试过程中插入备注


             Raises:
                 None: 此装饰器不抛出报错
                 """

    def log_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            with open(r'log\log_config.json', mode='r', encoding='utf') as config:
                log_path_data = json.load(config)
            file_path = log_path_data['json_log_path']
            log_dict = {"调用模块": module,
                        "调用函数": func.__name__,
                        "函数执行时间": f"{(end_time - start_time):.2f}秒",
                        "返回值": result
                        }
            if remarks is not None:
                log_dict["备注"] = remarks
            with open(file=file_path, mode='r', encoding='utf-8') as logging:
                log_data = json.load(logging)

            log_data.append(log_dict)

            with open(file=file_path, mode='w', encoding='utf-8') as logging:
                json.dump(log_data, logging, ensure_ascii=False, indent=4)

            return result

        return wrapper

    return log_decorator


def log4txt(module,
            remarks=None,
            numoflines=72,
            divisionline='-'):
    """
         一个用于记录日志的装饰器，接受日志目录和日志编写模式的参数输入，在日志中记录运行的函数和运行的时间，仅适用于返回单个值的函数

         Args:
             module(str):在面向对象编程中调用的模块的名称，方便开发进行调试时明确模块
             remarks(str):备注，便于开发者在调试过程中插入备注
             numoflines(int):个性化选择，在txt文件中分隔线中'-'的数量
             divisionline(str):个性化选择，在txt文件中自定义分割线的样式


         Raises:
             None: 此装饰器不抛出报错
             """

    def log_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            with open(r'log\log_config.json', mode='r', encoding='utf') as config:
                log_path_data = json.load(config)
            file_path = log_path_data['txt_log_path']
            if remarks is None:
                logging_list = [
                    f"调用模块：{module} \n",
                    f"调用函数：{func.__name__} \n",
                    f"函数执行时间：{(end_time - start_time):.2f}秒 \n",
                    f"返回值：{result} \n",
                    "\n",
                    divisionline * numoflines,
                    "\n"
                ]
            else:
                logging_list = [
                    f"调用模块：{module} \n",
                    f"调用函数：{func.__name__} \n",
                    f"函数执行时间：{(end_time - start_time):.2f}秒 \n",
                    f"返回值：{result} \n",
                    f"备注：{remarks}\n"
                    "\n",
                    divisionline * numoflines,
                    "\n"
                ]

            with open(file=file_path, mode='a', encoding='utf-8') as logging:
                for log in logging_list:
                    logging.write(log)

                return result

        return wrapper

    return log_decorator


def log4csv(module, remarks=None):
    """
             一个用于记录日志的装饰器，接受日志目录和日志编写模式的参数输入，在日志中记录运行的函数和运行的时间，仅适用于返回单个值的函数

             Args:
                 module(str):在面向对象编程中调用的模块的名称，方便开发进行调试时明确模块
                 remarks(str):备注，便于开发者在调试过程中插入备注


             Raises:
                 None: 此装饰器不抛出报错
                 """

    def log_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            with open(r'log\log_config.json', mode='r', encoding='utf') as config:
                log_path_data = json.load(config)
            file_path = log_path_data['csv_log_path']
            info_row = [module, func.__name__, f"{(end_time - start_time):.2f}秒", result, remarks]
            df = pd.read_csv(file_path)
            new_row = pd.DataFrame([info_row], columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(file_path, index=True)
            return result

        return wrapper

    return log_decorator


def log4excel(module, remarks=None):
    """
             一个用于记录日志的装饰器，接受日志目录和日志编写模式的参数输入，在日志中记录运行的函数和运行的时间，仅适用于返回单个值的函数

             Args:
                 module(str):在面向对象编程中调用的模块的名称，方便开发进行调试时明确模块
                 remarks(str):备注，便于开发者在调试过程中插入备注


             Raises:
                 None: 此装饰器不抛出报错
                 """

    def log_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            with open(r'log\log_config.json', mode='r', encoding='utf') as config:
                log_path_data = json.load(config)
            file_path = log_path_data['excel_log_path']
            info_row = [module, func.__name__, f"{(end_time - start_time):.2f}秒", result, remarks]
            df = pd.read_excel(file_path)
            new_row = pd.DataFrame([info_row], columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_excel(file_path, index=False)
            return result

        return wrapper

    return log_decorator


@log4txt("log", remarks="这是一次测试", divisionline="^", numoflines=10)
def test_func():
    return "测试函数被执行"


@log4json("log", remarks="这是一次测试")
def test_func1():
    return "测试函数被执行"


@log4csv("log", remarks="这是一次测试")
def test_func2():
    return "测试函数被执行"


@log4excel("log", remarks="这是一次测试")
def test_func3():
    return "测试函数被执行"


if __name__ == "__main__":
    logger_ini()
    test_func()
    test_func1()
    test_func2()
    test_func3()

