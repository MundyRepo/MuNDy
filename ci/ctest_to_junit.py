#!/usr/bin/env python3
"""Convert a CTest ``Test.xml`` into a JUnit XML file for Jenkins.

The TriBITS CTest -S driver already runs the tests and submits results to
CDash, writing a CTest ``Test.xml`` into ``build/Testing/<TAG>/``. CTest can
emit JUnit directly via ``ctest_test(OUTPUT_JUNIT ...)``, but the TriBITS driver
does not forward that option and there is no global CTest variable for it, so we
convert the already-written ``Test.xml`` rather than re-running the suite.

Usage:
    ctest_to_junit.py <Test.xml> <out.junit.xml>
"""
import sys
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape


def _named_measurement(test, name):
    for nm in test.findall("./Results/NamedMeasurement"):
        if nm.get("name") == name:
            value = nm.find("Value")
            return value.text if value is not None else None
    return None


def _stdout(test):
    measurement = test.find("./Results/Measurement/Value")
    return measurement.text if measurement is not None else ""


def convert(test_xml_path, junit_path):
    tree = ET.parse(test_xml_path)
    root = tree.getroot()  # <Site>
    site = root.get("Name", "")
    build = root.get("BuildName", "")

    tests = root.findall("./Testing/Test")
    failures = 0
    total_time = 0.0
    cases = []

    for test in tests:
        status = test.get("Status", "")
        name = test.findtext("Name", default="")
        time_str = _named_measurement(test, "Execution Time") or "0"
        try:
            t = float(time_str)
        except ValueError:
            t = 0.0
        total_time += t

        case = ['  <testcase classname="{}" name="{}" time="{:.3f}"'.format(
            escape(build or "ctest"), escape(name), t)]
        if status == "passed":
            case.append("/>")
        elif status == "notrun":
            case.append(">\n    <skipped/>\n  </testcase>")
        else:  # failed (or anything else) -> failure
            failures += 1
            reason = _named_measurement(test, "Exit Code") or status
            # Split any literal "]]>" so it cannot terminate our CDATA section
            # and corrupt the JUnit document.
            output = (_stdout(test) or "").replace("]]>", "]]]]><![CDATA[>")
            case.append('>\n    <failure message="{}"><![CDATA[{}]]></failure>\n  </testcase>'.format(
                escape(str(reason)), output))
        cases.append("".join(case))

    with open(junit_path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<testsuite name="{}" tests="{}" failures="{}" errors="0" time="{:.3f}">\n'.format(
            escape(site or "Mundy"), len(tests), failures, total_time))
        f.write("\n".join(cases))
        f.write("\n</testsuite>\n")

    print("Wrote {} ({} tests, {} failures) from {}".format(
        junit_path, len(tests), failures, test_xml_path))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write(__doc__)
        sys.exit(2)
    convert(sys.argv[1], sys.argv[2])
