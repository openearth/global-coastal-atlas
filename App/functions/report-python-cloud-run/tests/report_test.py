def test_generate_report_content():
    from report.report import (
        generate_report_content,
        POLYGON_DEFAULT,
        STAC_ROOT_DEFAULT,
    )
    from shapely.geometry import shape
    import json

    polygon = shape(json.loads(POLYGON_DEFAULT))
    report_content = generate_report_content(polygon, STAC_ROOT_DEFAULT)

    assert len(report_content.datasets) == 1


def test_generate_report_content_with_empty_polygon():
    from report.report import (
        generate_report_content,
        STAC_ROOT_DEFAULT,
    )
    from shapely.geometry import shape
    import json

    empty_polygon_str = """{
        "coordinates": [
          [
            [
              56.25212001959346,
              4.361498257654745
            ],
            [
              58.02992161118161,
              0.7766585367626817
            ],
            [
              64.18450423501352,
              3.346070943027584
            ],
            [
              59.40135742046908,
              7.242292017102514
            ],
            [
              56.25212001959346,
              4.361498257654745
            ]
          ]
        ],
        "type": "Polygon"
      }"""

    polygon = shape(json.loads(empty_polygon_str))
    report_content = generate_report_content(polygon, STAC_ROOT_DEFAULT)

    assert len(report_content.datasets) == 0
