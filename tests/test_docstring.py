import docstring_parser

doc = """角色扮演沉浸式时间推进器。当你要去做某事时（泡茶/咖啡、做饭、查资料、拿东西、学习、看网课等），用此工具推进时间。系统自动推进，不需要手动 advance。interact_prompt是用于在长任务中途主动和用户互动的一句指令（如"问口味偏好""吐槽这题好难""分享一下有趣的内容"），AI会根据这句指令自己构思怎么互动。

        Args:
            action(string): create创建 list查看 cancel取消。
            tasks_json(string): action=create时必填。JSON数组1个主任务: [{"name":"做什么","duration_minutes":3,"prompt":"到点状态提示","interact_prompt":"中途主动互动的一句指令(可选,如'问口味偏好')","interact_duration":0.5}]
            chain_id(string): action=cancel时必填。
        """

p = docstring_parser.parse(doc)
print(f"Params: {len(p.params)}")
for a in p.params:
    print(f"  {a.arg_name}: {a.type_name}")
