#!/usr/bin/env python3
"""
PDF Analysis Tool

Analyzes a PDF using Natural PDF's capabilities to understand what it can actually extract.
This shows what Natural PDF sees vs. what users are struggling with.

Usage:
    python pdf_analyzer.py path/to/document.pdf [num_pages] [output_folder]
"""

import json
import sys
from pathlib import Path

import natural_pdf as npdf
from natural_pdf.elements.element_collection import ElementCollection


def analyze_pdf(
    pdf_path, num_pages=1, output_folder="analysis_results", create_timestamp_folder=True
):
    """Analyze a PDF using Natural PDF's full capabilities"""

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ File not found: {pdf_path}")
        return

    # Create output folder structure
    base_output_dir = Path(output_folder)
    base_output_dir.mkdir(exist_ok=True)

    # If create_timestamp_folder=True, create a timestamped run folder for batch analysis
    if create_timestamp_folder:
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = base_output_dir / f"run_{timestamp}"
        run_output_dir.mkdir(exist_ok=True)
    else:
        run_output_dir = base_output_dir

    # Create subfolder for this specific PDF within the run folder
    pdf_output_dir = run_output_dir / pdf_file.stem
    pdf_output_dir.mkdir(exist_ok=True)

    print(f"🔍 ANALYZING: {pdf_file.name}")
    print(f"📁 Output folder: {pdf_output_dir}")
    print("=" * 80)

    analysis_data = {
        "pdf_name": pdf_file.name,
        "pdf_path": str(pdf_file),
        "analysis_timestamp": None,
        "pages": [],
    }

    try:
        # Load PDF
        pdf = npdf.PDF(str(pdf_file))
        total_pages = len(pdf.pages)
        pages_to_analyze = min(num_pages, total_pages)

        analysis_data["total_pages"] = total_pages
        analysis_data["pages_analyzed"] = pages_to_analyze

        print(f"📄 Total pages: {total_pages}")
        print(f"🔍 Analyzing first {pages_to_analyze} page(s)")
        print()

        for page_num in range(pages_to_analyze):
            page = pdf.pages[page_num]

            page_data = {
                "page_number": page_num + 1,
                "dimensions": {"width": page.width, "height": page.height},
                "describe": None,
                "extract_text": None,
                "extract_table": None,
                "analyze_layout": None,
                "regions": None,
                "elements_sample": None,
                "image_path": None,
            }

            print(f"📄 PAGE {page_num + 1}")
            print("-" * 60)

            # Basic page info
            print(f"📐 Dimensions: {page.width:.1f} x {page.height:.1f}")

            # 1. .describe() - Overview of elements
            print(f"\n🤖 PAGE.DESCRIBE():")
            try:
                description = page.describe()
                print(description)
                page_data["describe"] = str(description)

                # Save describe output to file
                with open(pdf_output_dir / f"page_{page_num + 1}_describe.txt", "w") as f:
                    f.write(str(description))

            except Exception as e:
                print(f"❌ describe() failed: {e}")
                page_data["describe"] = f"ERROR: {e}"

            # 2. .extract_text() - Raw text extraction
            print(f"\n📝 PAGE.EXTRACT_TEXT():")
            try:
                text = page.extract_text()
                if text:
                    print(f"Length: {len(text)} characters")
                    # Show first 300 chars
                    preview = text[:300].replace("\n", "\\n")
                    print(f"Preview: {preview}...")
                    page_data["extract_text"] = {
                        "length": len(text),
                        "preview": preview,
                        "full_text": text,
                    }

                    # Save full text to file
                    with open(pdf_output_dir / f"page_{page_num + 1}_text.txt", "w") as f:
                        f.write(text)

                else:
                    print("No text extracted")
                    page_data["extract_text"] = {"length": 0, "preview": "", "full_text": ""}
            except Exception as e:
                print(f"❌ extract_text() failed: {e}")
                page_data["extract_text"] = f"ERROR: {e}"

            # 3. .extract_table() - Table extraction (returns List[List[str]])
            print(f"\n📊 PAGE.EXTRACT_TABLE():")
            try:
                table_data = page.extract_table()  # This returns List[List[Optional[str]]]
                if table_data and len(table_data) > 0:
                    rows = len(table_data)
                    cols = len(table_data[0]) if table_data[0] else 0
                    print(f"Table found: {rows} rows x {cols} columns")
                    print("Sample data (first 3 rows):")
                    for i, row in enumerate(table_data[:3]):
                        print(f"  Row {i+1}: {row}")

                    page_data["extract_table"] = {
                        "found": True,
                        "rows": rows,
                        "columns": cols,
                        "data": table_data,
                    }

                    # Save table data as JSON
                    with open(pdf_output_dir / f"page_{page_num + 1}_table.json", "w") as f:
                        json.dump(table_data, f, indent=2)

                else:
                    print("No table extracted")
                    page_data["extract_table"] = {"found": False}
            except Exception as e:
                print(f"❌ extract_table() failed: {e}")
                page_data["extract_table"] = f"ERROR: {e}"

            # 4. .analyze_layout() - Layout analysis
            print(f"\n🏗️  PAGE.ANALYZE_LAYOUT():")
            try:
                layout = page.analyze_layout()
                if layout and len(layout) > 0:
                    print(f"Layout regions found: {len(layout)}")
                    layout_info = []
                    for i, region in enumerate(layout[:5]):  # Show first 5
                        region_info = {
                            "type": getattr(region, "type", "unknown"),
                            "bbox": [region.x0, region.top, region.x1, region.bottom],
                            "confidence": getattr(region, "confidence", 0),
                        }
                        layout_info.append(region_info)
                        print(
                            f"  {i+1}. {region_info['type']} at {region_info['bbox']} (conf: {region_info['confidence']:.2f})"
                        )

                    page_data["analyze_layout"] = {
                        "found": True,
                        "count": len(layout),
                        "regions": layout_info,
                    }
                else:
                    print("No layout regions found")
                    page_data["analyze_layout"] = {"found": False}
            except Exception as e:
                print(f"❌ analyze_layout() failed: {e}")
                page_data["analyze_layout"] = f"ERROR: {e}"

            # 4b. .analyze_layout('tatr') - Table structure analysis (append to preserve YOLO results)
            print(f"\n🏗️  PAGE.ANALYZE_LAYOUT('TATR') - Table Structure:")
            try:
                tatr_layout = page.analyze_layout("tatr", existing="append")
                if tatr_layout and len(tatr_layout) > 0:
                    print(f"TATR layout regions found: {len(tatr_layout)}")
                    tatr_info = []
                    for i, region in enumerate(tatr_layout[:5]):  # Show first 5
                        region_info = {
                            "type": getattr(region, "type", "unknown"),
                            "bbox": [region.x0, region.top, region.x1, region.bottom],
                            "confidence": getattr(region, "confidence", 0),
                        }
                        tatr_info.append(region_info)
                        print(
                            f"  {i+1}. {region_info['type']} at {region_info['bbox']} (conf: {region_info['confidence']:.2f})"
                        )

                    page_data["analyze_layout_tatr"] = {
                        "found": True,
                        "count": len(tatr_layout),
                        "regions": tatr_info,
                    }

                    # Save TATR layout analysis to file
                    tatr_summary = f"TATR Layout Analysis\n{'='*50}\n"
                    tatr_summary += f"Found {len(tatr_layout)} regions:\n\n"
                    for i, region_info in enumerate(tatr_info):
                        tatr_summary += f"{i+1}. {region_info['type']} at {region_info['bbox']} (conf: {region_info['confidence']:.2f})\n"

                    with open(pdf_output_dir / f"page_{page_num + 1}_tatr_layout.txt", "w") as f:
                        f.write(tatr_summary)

                    # Try to get detailed table structure
                    try:
                        table_structure = page.find_table_structure()
                        if table_structure:
                            print(f"Table structure found with {len(table_structure)} elements")
                            table_details = str(table_structure)
                            page_data["table_structure"] = {
                                "found": True,
                                "count": len(table_structure),
                                "details": table_details[:1000]
                                + ("..." if len(table_details) > 1000 else ""),
                            }

                            # Save table structure to file
                            with open(
                                pdf_output_dir / f"page_{page_num + 1}_table_structure.txt", "w"
                            ) as f:
                                f.write(table_details)
                        else:
                            page_data["table_structure"] = {"found": False}
                    except Exception as te:
                        print(f"Table structure detection failed: {te}")
                        page_data["table_structure"] = f"ERROR: {te}"
                else:
                    print("No TATR layout regions found")
                    page_data["analyze_layout_tatr"] = {"found": False}
                    page_data["table_structure"] = {"found": False}
            except Exception as e:
                print(f"❌ analyze_layout('tatr') failed: {e}")
                page_data["analyze_layout_tatr"] = f"ERROR: {e}"
                page_data["table_structure"] = f"ERROR: {e}"

            # 5. Find regions by model and save separate + combined files
            print(f"\n📍 REGION ANALYSIS - By Model:")
            try:
                all_regions = page.find_all("region")
                if all_regions and len(all_regions) > 0:
                    print(f"Total regions found: {len(all_regions)}")

                    # Group regions by model/source
                    yolo_regions = [
                        r
                        for r in all_regions
                        if getattr(r, "model", "") == "" or getattr(r, "model", "") == "yolo"
                    ]
                    tatr_regions = [r for r in all_regions if getattr(r, "model", "") == "tatr"]
                    other_regions = [
                        r
                        for r in all_regions
                        if getattr(r, "model", "") not in ["", "yolo", "tatr"]
                    ]

                    print(f"  YOLO regions: {len(yolo_regions)}")
                    print(f"  TATR regions: {len(tatr_regions)}")
                    print(f"  Other regions: {len(other_regions)}")

                    # Save separate files for each model
                    if yolo_regions:
                        yolo_inspect = str(ElementCollection(yolo_regions).inspect(limit=1000))
                        with open(
                            pdf_output_dir / f"page_{page_num + 1}_yolo_regions.txt", "w"
                        ) as f:
                            f.write(
                                f"YOLO Layout Regions ({len(yolo_regions)} found)\n{'='*50}\n\n{yolo_inspect}"
                            )

                    if tatr_regions:
                        tatr_inspect = str(ElementCollection(tatr_regions).inspect(limit=1000))
                        with open(
                            pdf_output_dir / f"page_{page_num + 1}_tatr_regions.txt", "w"
                        ) as f:
                            f.write(
                                f"TATR Layout Regions ({len(tatr_regions)} found)\n{'='*50}\n\n{tatr_inspect}"
                            )

                    # Combined regions inspect
                    all_inspect = str(all_regions.inspect(limit=1000))
                    print(f"Combined regions preview (first 500 chars):\n{all_inspect[:500]}...")

                    # Save combined regions file
                    with open(pdf_output_dir / f"page_{page_num + 1}_all_regions.txt", "w") as f:
                        f.write(f"All Layout Regions ({len(all_regions)} found)\n{'='*50}\n")
                        f.write(
                            f"YOLO: {len(yolo_regions)}, TATR: {len(tatr_regions)}, Other: {len(other_regions)}\n\n"
                        )
                        f.write(all_inspect)

                    page_data["regions"] = {
                        "found": True,
                        "total_count": len(all_regions),
                        "yolo_count": len(yolo_regions),
                        "tatr_count": len(tatr_regions),
                        "other_count": len(other_regions),
                        "inspect_preview": (
                            all_inspect[:500] + "..." if len(all_inspect) > 500 else all_inspect
                        ),
                    }

                else:
                    print("No regions found")
                    page_data["regions"] = {"found": False}
            except Exception as e:
                print(f"❌ region analysis failed: {e}")
                page_data["regions"] = f"ERROR: {e}"

            # 6. General element inspection
            print(f"\n🔍 GENERAL ELEMENT INSPECTION:")
            try:
                # Count different element types
                all_elements = page.find_all("*")
                if all_elements and len(all_elements) > 0:
                    print(f"Total elements: {len(all_elements)}")

                    # Full inspect output - shows complete breakdown
                    print(f"\nFull element breakdown (.inspect()):")
                    # Get string representation of inspect result (increased limit)
                    inspect_result = all_elements.inspect(limit=1000)
                    inspect_text = str(inspect_result)
                    print(inspect_text)

                    # Sample some elements for detailed inspection
                    sample_elements = all_elements[:10]  # First 10 elements
                    print(f"Sample of first 10 elements:")
                    elements_sample = []
                    for i, elem in enumerate(sample_elements):
                        elem_type = getattr(elem, "object_type", "unknown")
                        text_preview = (
                            getattr(elem, "text", "")[:30] if hasattr(elem, "text") else ""
                        )
                        elem_info = {
                            "type": elem_type,
                            "text": text_preview,
                            "x0": elem.x0,
                            "top": elem.top,
                        }
                        elements_sample.append(elem_info)
                        print(
                            f"  {i+1}. {elem_type}: '{text_preview}' at ({elem.x0:.0f}, {elem.top:.0f})"
                        )

                    page_data["elements_sample"] = {
                        "total_count": len(all_elements),
                        "full_inspect": inspect_text,
                        "sample": elements_sample,
                    }

                    # Save full inspect to file
                    with open(
                        pdf_output_dir / f"page_{page_num + 1}_all_elements_inspect.txt", "w"
                    ) as f:
                        f.write(inspect_text)

                else:
                    print("No elements found")
                    page_data["elements_sample"] = {"total_count": 0, "sample": []}
            except Exception as e:
                print(f"❌ element inspection failed: {e}")
                page_data["elements_sample"] = f"ERROR: {e}"

            # 7. Render page as image
            print(f"\n🖼️  RENDERING PAGE AS IMAGE:")
            try:
                img = page.render(resolution=144)
                print(f"Image: {img.width}x{img.height} pixels")

                # Save image in output folder
                img_filename = f"page_{page_num + 1}.png"
                img_path = pdf_output_dir / img_filename
                img.save(str(img_path))
                print(f"Saved: {img_path}")
                page_data["image_path"] = str(img_path)

            except Exception as e:
                print(f"❌ image rendering failed: {e}")
                page_data["image_path"] = f"ERROR: {e}"

            analysis_data["pages"].append(page_data)

            if page_num < pages_to_analyze - 1:
                print("\n" + "=" * 80 + "\n")

        # Save complete analysis data as JSON
        import datetime

        analysis_data["analysis_timestamp"] = datetime.datetime.now().isoformat()

        summary_file = pdf_output_dir / "analysis_summary.json"
        with open(summary_file, "w") as f:
            json.dump(analysis_data, f, indent=2)

        print(f"\n✅ ANALYSIS COMPLETE")
        print(f"📊 Summary: Analyzed {pages_to_analyze} page(s) of {pdf_file.name}")
        print(f"📁 All results saved to: {pdf_output_dir}")
        print(f"📋 Summary JSON: {summary_file}")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print(
            "Usage: python pdf_analyzer.py <pdf_path> [num_pages] [output_folder] [--no-timestamp]"
        )
        print("Example: python pdf_analyzer.py bad-pdfs/submissions/Focus.pdf 2 analysis_results")
        print("         python pdf_analyzer.py Focus.pdf 1 my_analysis --no-timestamp")
        sys.exit(1)

    pdf_path = sys.argv[1]
    num_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    output_folder = "analysis_results"
    create_timestamp_folder = True

    # Parse remaining arguments
    for arg in sys.argv[3:]:
        if arg == "--no-timestamp":
            create_timestamp_folder = False
        elif not arg.startswith("--"):
            output_folder = arg

    analyze_pdf(pdf_path, num_pages, output_folder, create_timestamp_folder)


if __name__ == "__main__":
    main()
