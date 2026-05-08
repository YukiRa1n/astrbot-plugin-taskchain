import docstring_parser

doc = """角色扮演沉浸式时间推进器。当你表示要去做某事（泡茶/咖啡、做饭、查资料、拿东西等），或用户说一会去做什么时，用此工具推进时间。日常1-5分钟，学习/看网课20-45分钟。interact是期间提示词不是逐字稿。调用即进入状态，不在回复中提工具或时间。

        Args:
            action(string): create创建 list查看 cancel取消 advance手动推进。
            tasks_json(string): action=create时必填。JSON数组1个主任务: [{"name":"做什么","duration_minutes":3,"prompt":"到点状态提示","interact":"期间互动提示词(可选)","interact_duration":0.5}]
            chain_id(string): action=cancel/advance时必填。
        """

p = docstring_parser.parse(doc)
print(f"Params: {len(p.params)}")
for a in p.params:
    print(f"  {a.arg_name}: {a.type_name}")
