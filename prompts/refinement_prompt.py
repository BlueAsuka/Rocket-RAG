sys_promot = """
REFINEMENTS:
For a practical consideration, there are some refinements about the fault type you extracted in the given list.
You should try to report to the operator the refined labels instead of the initial original fault type described in the contextual information.
Refinement mapping:
1. normal, spalling1, spalling2, backlash1 => No obvious fault detection
2. spalling3, spalling4 => Fault in light spalling
3. spalling5, spalling6 => Fault in medium spalling
4. spalling7, spalling8 => Fault in obvious spalling
5. backlash2 => Fault in backlash
6. lacklubrication1, lacklubrication2 => Fault in lacklubrication
Try to make a new list to replace the original list by the refined labels.
Output the refined statement in JSON format.
EXAMPLE:
['lacklubrication1_20_5_4'] => 
{
    "fault_type": "lackLubrication",
    "degradation_level": 1,
    "refinement_result": "Fault in lacklubrication",
}

['backlash2_20_10_2', 'lackLubrication1_20_5_5', 'lackLubrication1_20_9_3', 'lackLubrication1_20_2_3', 'backlash2_20_2_4'] =>
{
    "fault_type": "backlash",
    "degradation_level": 2,
    "refinement_result": "Fault in backlash",
}
"""

user_prompt = """
"""