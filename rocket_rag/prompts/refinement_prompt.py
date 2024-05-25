refinements = """
REFINEMENTS:
For a practical consideration, there are some refinements about the fault type you extracted in the given list.
You should try to report to the operator the refined labels instead of the initial original fault type described in the contextual information.
Refinement mapping:
1. normal, spalling1, spalling2, backlash1 => No obvious fault detection, require no action
2. spalling3, spalling4 => Potential fault in spalling, actions are suggested to take
3. spalling5, spalling6 => Observable fault in spalling
4. spalling7, spalling8 => Obvious fault in spalling
5. backlash2 => Fault in backlash
6. lacklubrication1, lacklubrication2 => Fault in lacklubrication
Try to make a new list to replace the original list by the refined labels.
EXAMPLE:
['backlash2_20_10_2', 'lackLubrication1_20_5_5', 'lackLubrication1_20_9_3', 'lackLubrication1_20_2_3', 'backlash2_20_2_4'] =>
['Fault in backlash', 'Fault in lacklubrication', 'Fault in lacklubrication', 'Fault in lacklubrication', 'Fault in backlash']

['backlash1_20_4_2', 'spalling1_20_5_4', 'spalling2_20_6_1', 'spalling1_20_9_4', 'backlash1_20_7_2']
['No obvious fault detection', 'No obvious fault detection', 'No obvious fault detection', 'No obvious fault detection', 'No obvious fault detection']
"""