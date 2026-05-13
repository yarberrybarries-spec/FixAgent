"""
意图识别测试脚本

使用方法：
1. 确保 .env 配置正确（DASHSCOPE_API_KEY）
2. 运行脚本：python test_intention_recognizer.py
3. 输入测试语句查看意图识别结果
4. 输入 q 退出
"""

import asyncio
from agents.intention import get_intention_recognizer
from schemas.models import IntentionType


def print_result(result):
    """格式化打印识别结果"""
    print("=" * 50)
    print(f"意图类型: {result.intention.value}")
    print(f"置信度: {result.confidence:.2f}")
    print(f"识别理由: {result.reasoning}")
    print("=" * 50)


# 预置测试用例
TEST_CASES = {
    "troubleshoot": [
        "轴承过热是什么原因",
        "电动机不转了怎么回事",
        "为什么会有异响",
        "发动机熄火了什么原因",
    ],
    "seek_guidance": [
        "怎么维修这台设备",
        "设备坏了怎么修",
        "操作步骤是什么",
        "告诉我怎么加油保养",
        "拆卸轴承的正确方法",
    ],
    "query_knowledge": [
        "什么是轴承",
        "电动机的工作原理是什么",
        "请介绍一下这台设备的规格",
    ],
    "submit_case": [
        "提交一个维修案例",
        "上传这次的故障记录",
        "分享这次的经验",
    ],
    "general_chat": [
        "今天天气不错",
        "你好",
        "谢谢",
    ],
}


async def main():
    print("=" * 50)
    print("  意图识别测试工具")
    print("=" * 50)
    print()

    recognizer = get_intention_recognizer()

    while True:
        print("\n请选择操作：")
        print("  1. 手动输入测试")
        print("  2. 运行预置测试用例")
        print("  q. 退出")
        choice = input("\n请输入选项: ").strip()

        if choice == "q":
            print("退出测试")
            break

        elif choice == "1":
            print("\n" + "=" * 50)
            print("  手动输入模式")
            print("=" * 50)
            print("输入语句进行测试（输入 q 返回上级菜单）")
            print()

            while True:
                message = input("请输入测试语句: ").strip()

                if message.lower() == "q":
                    break

                if not message:
                    print("输入不能为空，请重新输入")
                    continue

                print("\n正在识别...")
                try:
                    result = await recognizer.recognize(message)
                    print_result(result)
                except Exception as e:
                    print(f"识别失败: {e}")

        elif choice == "2":
            print("\n" + "=" * 50)
            print("  预置测试用例")
            print("=" * 50)
            print()

            for intention, cases in TEST_CASES.items():
                print(f"\n【{intention}】")
                for i, case in enumerate(cases, 1):
                    print(f"  {i}. {case}")

            print("\n" + "-" * 50)
            print("开始运行预置测试...")
            print("-" * 50)

            all_passed = True
            for intention_name, cases in TEST_CASES.items():
                expected = IntentionType(intention_name)
                print(f"\n>>> 测试 {intention_name} 类别:")

                for case in cases:
                    try:
                        result = await recognizer.recognize(case)
                        status = "✓" if result.intention == expected else "✗"
                        if result.intention != expected:
                            all_passed = False
                        print(f"  {status} 输入: {case}")
                        print(f"    识别结果: {result.intention.value} (置信度: {result.confidence:.2f})")
                        if result.intention != expected:
                            print(f"    期望: {expected.value}, 实际: {result.intention.value}")
                    except Exception as e:
                        print(f"  ✗ 输入: {case}")
                        print(f"    错误: {e}")
                        all_passed = False

            print("\n" + "=" * 50)
            if all_passed:
                print("  所有测试用例通过！")
            else:
                print("  部分测试用例未通过")
            print("=" * 50)

        else:
            print(f"无效选项: {choice}")


if __name__ == "__main__":
    asyncio.run(main())
