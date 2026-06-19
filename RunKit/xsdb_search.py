# based on https://github.com/cms-sw/xsecdb/blob/master/scripts/xsdb_search.py

import io
import json
import os
import pycurl
import sys

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(file_dir))
    __package__ = "RunKit"

from .run_tools import ps_call


def search_xsdb(query_dict):
    base_url = "https://cms-gen-dev.cern.ch/xsdb"
    api_url = base_url + "/api"
    query = {
        "search": query_dict,
        "pagination": {"pageSize": 0, "currentPage": 0},
        "orderBy": {},
    }
    query_str = json.dumps(query)
    home = os.environ["HOME"]
    cookie = os.path.join(home, "private/xsdbdev-cookie.txt")
    ps_call(
        ["cern-get-sso-cookie", "-u", base_url, "--krb", "-r", "-o", cookie],
        env={"KRB5CCNAME": os.environ["KRB5CCNAME"]},
    )

    bytes_io = io.BytesIO()
    c = pycurl.Curl()
    c.setopt(pycurl.FOLLOWLOCATION, 1)
    c.setopt(pycurl.COOKIEJAR, cookie)
    c.setopt(pycurl.COOKIEFILE, cookie)
    c.setopt(
        pycurl.HTTPHEADER, ["Content-Type:application/json", "Accept:application/json"]
    )
    c.setopt(pycurl.VERBOSE, 0)
    c.setopt(pycurl.URL, api_url + "/search")
    c.setopt(pycurl.WRITEFUNCTION, bytes_io.write)
    c.setopt(pycurl.POST, 1)
    c.setopt(pycurl.POSTFIELDS, query_str)
    c.perform()

    out = bytes_io.getvalue().decode("utf-8")
    return json.loads(out)


if __name__ == "__main__":
    query_dict = {}
    for line in sys.argv[1:]:
        key, val = line.split("=")
        query_dict[key] = val
    print(json.dumps(search_xsdb(query_dict), indent=2))
