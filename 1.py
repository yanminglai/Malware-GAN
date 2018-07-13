import CuckooAPI
api = CuckooAPI.CuckooAPI("10.0.0.144", APIPY=True, port=8090)
api.submitfile("malware.exe")