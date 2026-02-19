from Segments.EquitySegment import EquitySegment
import json

es = EquitySegment()
costs = es.calculate_etf_delivery_costs('buy', 100000, 0.1)
print(f"RAW COSTS DICT: {json.dumps(costs, indent=2)}")

b = costs.get('brokerage', 0)
tc = (
    costs.get('stt', 0) + 
    costs.get('stamp_duty', 0) + 
    costs.get('exchange_charges', 0) + 
    costs.get('sebi_charges', 0) + 
    costs.get('gst', 0)
)
print(f"Calculated b: {b}")
print(f"Calculated tc: {tc}")
print(f"Sum (b + tc): {b + tc}")
print(f"Key 'total_costs' in dict: {costs.get('total_costs', 'MISSING')}")
print(f"Equality check: {round(b + tc, 2) == round(costs.get('total_costs', 0), 2)}")
