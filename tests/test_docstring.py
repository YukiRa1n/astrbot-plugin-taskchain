import docstring_parser

doc = """角色扮演沉浸式时间推进器。用户说一会去做什么时设置任务，自动插入互动环节让等待感消失。调用即进入状态，不在回复中提工具或时间。日常小事设1-5分钟，长时间活动(学习/看网课)可20-45分钟。interact可写问问题、吐槽、抱怨，让对话生动。

        Args:
            action(string): 操作类型: create创建 list查看 cancel取消 advance手动推进。
            tasks_json(string): action=create时必填。JSON数组放1个主任务: [{"name":"做什么","duration_minutes":3,"prompt":"到点状态提示","interact":"期间想说的话(可选)","interact_duration":0.5}]
            chain_id(string): action=cancel/advance时必填。
        """

parsed = docstring_parser.parse(doc)
print("Description:", parsed.description[:60] if parsed.description else "NONE")
print("Params count:", len(parsed.params))
for p in parsed.params:
    print(f"  {p.arg_name}: type={p.type_name}")
