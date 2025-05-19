import pygame

# 初始化 pygame
pygame.init()

# 检查是否有手柄连接
if pygame.joystick.get_count() == 0:
    print("未检测到手柄，请连接手柄后重试！")
    exit()

# 初始化手柄
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"已连接手柄：{joystick.get_name()}")

def main():
    """主循环，用于读取手柄摇杆的输入"""
    try:
        while True:
            # 处理事件队列
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            # 获取摇杆的轴数据（通常摇杆有X和Y两个轴）
            axis_x = joystick.get_axis(2)  # X轴
            axis_y = joystick.get_axis(3)  # Y轴

            # 打印摇杆的实时位置
            print(f"摇杆位置 - X: {axis_x:.2f}, Y: {axis_y:.2f}")

    except KeyboardInterrupt:
        print("程序已退出。")

if __name__ == "__main__":
    main()