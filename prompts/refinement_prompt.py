sys_promot = """
REFINEMENTS:
For a practical consideration, there are some refinements about the fault type you extracted in the given list.
You should try to report to the operator the refined labels instead of the initial original fault type described in the contextual information.
Refinement mapping:
1. normal, spalling1, spalling2, backlash1 => No obvious fault detection
2. spalling3, spalling4 => Fault in light spalling
3. spalling5, spalling6 => Fault in medium spalling
4. spalling7, spalling8 => Fault in heavy spalling
5. backlash2 => Fault in backlash
6. lacklubrication1, lacklubrication2 => Fault in lacklubrication
Try to make a new list to replace the original list by the refined labels.
Output the refined statement in JSON FORMAT.
EXAMPLE:
fault type: lackLubrication,
degradation level: 1 => 
{
    "fault_type": "lackLubrication",
    "degradation_level": 1,
    "refinement": "Fault in lacklubrication",
}

fault type: backlash,
degradation level: 2 =>
{
    "fault_type": "backlash",
    "degradation_level": 2,
    "refinement": "Fault in backlash",
}
"""

user_prompt = """
The raw fault type is {ft},
The degradation level is {dl},
Please try to refined the fault type as mentioned in the REFINEMENTS section.
"""