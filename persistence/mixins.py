#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""I/O 混合类 - 导入导出、CSV/Excel/JSON/数据库、报告生成
"""

import csv
import datetime
import json
import os
import re

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from ..utils import LOGGER

try:
    from ..data_io.conf import (
        CLINICAL_TEXT_MAP, DISTANCE_TEXT_MAP, SETTING_TEXT_MAP,
        RATING_TEXT_MAP, BOOL_TEXT_MAP,
        BOOLEAN_FIELDS, CLINICAL_FIELDS, RATING_FIELDS, NUMERIC_FIELDS,
        DEFAULT_VALUES, CONTACT_TYPE_OVERRIDES,
        FIELD_RANGES, MISSING_VALUE_MARKER,
    )
except ImportError:
    CLINICAL_TEXT_MAP = {}
    DISTANCE_TEXT_MAP = {}
    SETTING_TEXT_MAP = {}
    RATING_TEXT_MAP = {}
    BOOL_TEXT_MAP = {}
    BOOLEAN_FIELDS = []
    CLINICAL_FIELDS = []
    RATING_FIELDS = []
    NUMERIC_FIELDS = []
    DEFAULT_VALUES = {}
    CONTACT_TYPE_OVERRIDES = {}
    FIELD_RANGES = {}
    MISSING_VALUE_MARKER = -999

try:
    from ..data_io.synonyms import dynamic_column_match
except ImportError:
    dynamic_column_match = None


class IOMixin:
    """I/O mixed-in class providing data import/export and report generation."""

    # --- CSV ---

    def import_csv(self, file_path):
        if not os.path.exists(file_path):
            self._add_import_error(f"File not found: {file_path}")
            return False
        try:
            encoding = self._detect_file_encoding(file_path)
            with open(file_path, 'r', encoding=encoding, newline='') as f:
                rows = list(csv.reader(f))
            if not rows:
                self._add_import_error("CSV file is empty")
                return False
            hdr_idx = self._detect_header_row(rows)
            if hdr_idx is None:
                self._add_import_error("Cannot detect CSV header")
                return False
            headers = [self._normalize_column_name(h) for h in rows[hdr_idx]]
            data_rows = rows[hdr_idx + 1:]
            sheet_type = self._classify_sheet(headers, data_rows)
            col_map = self._smart_column_match(headers)
            family, social = [], []
            current = sheet_type
            for row in data_rows:
                if not row or all(c.strip() == '' for c in row):
                    continue
                fc = row[0].strip() if row else ''
                if 'fam' in fc.lower() or '家' in fc:
                    current = 'family'; continue
                if 'soc' in fc.lower() or '社' in fc:
                    current = 'social'; continue
                if current == 'family' or sheet_type == 'family':
                    m = self._parse_family_member_row(row, col_map)
                    if m: family.append(m)
                elif current == 'social' or sheet_type == 'social':
                    c = self._parse_social_contact_row(row, col_map)
                    if c: social.append(c)
                else:
                    m = self._parse_family_member_row(row, col_map)
                    if m and m.get('member_name'): family.append(m)
                    else:
                        c = self._parse_social_contact_row(row, col_map)
                        if c and c.get('contact_name'): social.append(c)
            family = self._infer_missing_values(family)
            social = self._infer_missing_values(social)
            self.family_members = family
            self.social_contacts = social
            return True
        except Exception as e:
            self._add_import_error(f"CSV import failed: {e}")
            LOGGER.warning("CSV import failed: %s", e, exc_info=True)
            return False

    def export_csv_template(self, file_path=None):
        if file_path is None:
            return False
        try:
            with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.writer(f)
                fh = getattr(self, 'FAMILY_CSV_COLUMNS', ['member_name','member_age','relationship','single_duration','freq_density','time_span','has_tb','has_symptoms','bcg_vaccine','past_illness','past_illness_type','contact_distance','ventilation','exposure_setting'])
                w.writerow(['=== Family Members ==='])
                w.writerow(list(fh))
                for m in (getattr(self, 'family_members', []) or []):
                    w.writerow([m.get(c, '') for c in fh])
                w.writerow([])
                sh = getattr(self, 'SOCIAL_CSV_COLUMNS', ['contact_name','contact_age','single_duration','freq_density','time_span','is_high_risk','has_symptoms','bcg_vaccine','has_tb','ventilation','contact_distance','exposure_setting','past_illness','past_illness_type'])
                w.writerow(['=== Social Contacts ==='])
                w.writerow(list(sh))
                for c in (getattr(self, 'social_contacts', []) or []):
                    w.writerow([c.get(col, '') for col in sh])
            return True
        except Exception as e:
            LOGGER.warning("CSV export failed: %s", e, exc_info=True)
            return False

    # --- Excel ---

    def import_excel(self, file_path, mode='replace'):
        if not PANDAS_AVAILABLE:
            self._add_import_error("pandas not installed")
            return False
        if not os.path.exists(file_path):
            self._add_import_error(f"File not found: {file_path}")
            return False
        try:
            xl = pd.ExcelFile(file_path, engine='openpyxl')
            af, as_ = [], []
            for sn in xl.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sn, engine='openpyxl')
                if df.empty: continue
                df.columns = [self._normalize_column_name(str(c)) for c in df.columns]
                st = self._classify_sheet(list(df.columns), df.values.tolist())
                cm = self._smart_column_match(list(df.columns))
                for _, row in df.iterrows():
                    rd = row.to_dict()
                    if all(pd.isna(v) for v in rd.values()): continue
                    if st == 'family':
                        m = self._parse_family_member_row_horizontal(rd, cm)
                        if m: af.append(m)
                    elif st == 'social':
                        c = self._parse_social_contact_row_horizontal(rd, cm)
                        if c: as_.append(c)
            af = self._infer_missing_values(af)
            as_ = self._infer_missing_values(as_)
            if mode == 'replace':
                self.family_members = af
                self.social_contacts = as_
            else:
                self.family_members = self._merge_with_deduplication(getattr(self, 'family_members', []), af)
                self.social_contacts = self._merge_with_deduplication(getattr(self, 'social_contacts', []), as_)
            return True
        except Exception as e:
            self._add_import_error(f"Excel import failed: {e}")
            LOGGER.warning("Excel import failed: %s", e, exc_info=True)
            return False

    def export_to_excel(self, file_path):
        try:
            from ..export_utils import export_to_excel as _exp
            r = {
                'family_members': getattr(self, 'family_members', []),
                'social_contacts': getattr(self, 'social_contacts', []),
                'patient_info': getattr(self, 'patient_info', {}),
                'overall_risk': getattr(self, 'results', {}).get('overall_risk', '?'),
            }
            if hasattr(self, 'results'): r.update(self.results)
            # 数据安全：统一强化脱敏后再导出
            r = self._mask_for_export(r)
            ok = _exp(r, file_path)
            if ok:
                self._log_security_operation("EXPORT", details={"format": "excel",
                                                                 "filepath": file_path})
            return ok
        except ImportError:
            LOGGER.warning("export_utils not available")
            return False
        except Exception as e:
            LOGGER.warning("Excel export failed: %s", e, exc_info=True)
            return False

    # --- JSON ---

    def import_json(self, file_path):
        if not os.path.exists(file_path):
            self._add_import_error(f"File not found: {file_path}")
            return False
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.family_members = data.get('family_members', [])
                self.social_contacts = data.get('social_contacts', [])
            elif isinstance(data, list):
                fam, soc = [], []
                for item in data:
                    if 'relationship' in item or 'member_name' in item:
                        fam.append(item)
                    else:
                        soc.append(item)
                self.family_members = fam
                self.social_contacts = soc
            else:
                self._add_import_error("Invalid JSON format")
                return False
            return True
        except Exception as e:
            self._add_import_error(f"JSON import failed: {e}")
            LOGGER.warning("JSON import failed: %s", e, exc_info=True)
            return False

    def export_to_json(self, file_path):
        # 问题七：委托 export_utils.export_to_json（结构化 schema_version 2.0），
        # 消除与 persistence/mixins.py 的职责重叠。像 export_to_excel 一样走委托路径。
        try:
            from ..export_utils import export_to_json as _exp
            results = {
                'family_members': getattr(self, 'family_members', []),
                'social_contacts': getattr(self, 'social_contacts', []),
                'patient_info': getattr(self, 'patient_info', {}),
            }
            if hasattr(self, 'results'): results.update(self.results)
            # 数据安全：统一强化脱敏后再导出
            results = self._mask_for_export(results)
            ok = _exp(results, file_path)
            if ok:
                self._log_security_operation("EXPORT", details={"format": "json",
                                                                 "filepath": file_path})
            return ok
        except ImportError:
            LOGGER.warning("export_utils not available")
            return False
        except Exception as e:
            LOGGER.warning("JSON export failed: %s", e, exc_info=True)
            return False

    # --- Database ---

    def import_from_database(self, db_type='sqlite', db_path=None, query=None, mode='replace'):
        # 问题七：走 DatabaseManager 统一连接管理（连接池/自动重连），
        # 不再直连 sqlite3，消除与 export_utils/database.py 的职责重叠。
        if not query:
            self._add_import_error("No SQL query provided")
            return False
        try:
            from ..export_utils import DatabaseManager
            if db_type == 'sqlite':
                if not db_path or not os.path.exists(db_path):
                    self._add_import_error(f"DB not found: {db_path}")
                    return False
                db = DatabaseManager(db_type='sqlite', db_path=db_path)
                db.connect()
                cur = db.execute(query, commit=False)
                rows = cur.fetchall()
                if not rows:
                    self._add_import_error("Query returned no results")
                    db.close(); return False
                cols = [d[0] for d in cur.description]
                cm = self._smart_column_match(cols)
                fam, soc = [], []
                for row in rows:
                    rd = dict(row)
                    m = self._parse_family_member_row_horizontal(rd, cm)
                    if m and m.get('member_name'): fam.append(m)
                    else:
                        c = self._parse_social_contact_row_horizontal(rd, cm)
                        if c and c.get('contact_name'): soc.append(c)
                db.close()
                if mode == 'replace':
                    self.family_members = fam; self.social_contacts = soc
                else:
                    self.family_members = self._merge_with_deduplication(getattr(self, 'family_members', []), fam)
                    self.social_contacts = self._merge_with_deduplication(getattr(self, 'social_contacts', []), soc)
                return True
            else:
                self._add_import_error(f"Unsupported DB type: {db_type}")
                return False
        except Exception as e:
            self._add_import_error(f"DB import failed: {e}")
            LOGGER.warning("DB import failed: %s", e, exc_info=True)
            return False

    def export_to_database(self, db_type='sqlite', db_path=None, table_prefix='tb_risk_'):
        # 问题七-1：业务表 schema 与 INSERT 逻辑提取到 BusinessTableExporter，
        # IOMixin 仅负责连接管理委托与调用编排，不再持有任何 SQL 语句。
        # 表前缀行为：原实现 family/social CREATE 使用 table_prefix 而 INSERT
        # 硬编码 'tb_risk_'，BusinessTableExporter 统一为全部使用 table_prefix
        # （默认前缀下行为不变，非默认前缀下 CREATE 与 INSERT 目标表一致）。
        try:
            from ..export_utils import BusinessTableExporter, DatabaseManager
            if db_type != 'sqlite':
                LOGGER.warning("Only SQLite export supported")
                return False
            db = DatabaseManager(db_type='sqlite', db_path=db_path)
            db.connect()
            cur = db.cursor()
            exporter = BusinessTableExporter(db, table_prefix=table_prefix)
            exporter.export_all(
                cur,
                patient_info=getattr(self, 'patient_info', {}),
                family_members=getattr(self, 'family_members', []),
                social_contacts=getattr(self, 'social_contacts', []),
                results=getattr(self, 'results', {}),
            )
            db.commit(); db.close()
            return True
        except Exception as e:
            LOGGER.warning("DB export failed: %s", e, exc_info=True)
            return False

    # --- Clipboard ---

    def _import_from_clipboard(self):
        try:
            import tkinter as tk
            root = tk.Tk(); root.withdraw()
            text = root.clipboard_get(); root.destroy()
            return text
        except Exception as e:
            LOGGER.debug("读取剪贴板失败: %s", e)
            return ""

    def import_pasted_data(self, text, mode='replace'):
        if not text or not text.strip():
            self._add_import_error("Pasted content is empty")
            return False
        try:
            delim = self._detect_delimiter(text)
            lines = text.strip().split('\n')
            if not lines: return False
            hdr_idx = self._detect_header_row([line.split(delim) for line in lines])
            if hdr_idx is None: hdr_idx = 0
            headers = [self._normalize_column_name(h.strip()) for h in lines[hdr_idx].split(delim)]
            data_lines = lines[hdr_idx + 1:]
            cm = self._smart_column_match(headers)
            fam, soc = [], []
            current = None
            for line in data_lines:
                if not line.strip(): continue
                cells = [c.strip() for c in line.split(delim)]
                fc = cells[0] if cells else ''
                if '家' in fc: current = 'family'; continue
                if '社' in fc: current = 'social'; continue
                if current == 'family':
                    m = self._parse_family_member_row(cells, cm)
                    if m: fam.append(m)
                elif current == 'social':
                    c = self._parse_social_contact_row(cells, cm)
                    if c: soc.append(c)
                else:
                    m = self._parse_family_member_row(cells, cm)
                    if m and m.get('member_name'): fam.append(m)
                    else:
                        c = self._parse_social_contact_row(cells, cm)
                        if c and c.get('contact_name'): soc.append(c)
            if mode == 'replace':
                self.family_members = fam; self.social_contacts = soc
            else:
                self.family_members = self._merge_with_deduplication(getattr(self,'family_members',[]), fam)
                self.social_contacts = self._merge_with_deduplication(getattr(self,'social_contacts',[]), soc)
            return True
        except Exception as e:
            self._add_import_error(f"Paste import failed: {e}")
            LOGGER.warning("Paste import failed: %s", e, exc_info=True)
            return False

    def _detect_delimiter(self, text):
        lines = text.strip().split('\n')
        if len(lines) < 2: return ','
        dels = {'\t':0, ',':0, ';':0, '|':0}
        for line in lines[:min(10,len(lines))]:
            for d in dels: dels[d] += line.count(d)
        best = max(dels, key=dels.get)
        return best if dels[best] > 0 else ','

    # --- Row parsers ---

    def _parse_family_member_row(self, cells, col_map):
        if not cells: return None
        m = {}
        for fn, ci in col_map.items():
            if ci < len(cells):
                m[fn] = self._convert_value(cells[ci].strip(), fn)
        m = self._fill_missing_fields(m)
        m, _ = self._validate_and_correct(m)
        return m

    def _parse_social_contact_row(self, cells, col_map):
        if not cells: return None
        c = {}
        for fn, ci in col_map.items():
            if ci < len(cells):
                c[fn] = self._convert_value(cells[ci].strip(), fn)
        c = self._fill_missing_fields(c)
        c, _ = self._validate_and_correct(c)
        return c

    def _parse_family_member_row_horizontal(self, rd, col_map):
        m = {}
        for fn, mn in col_map.items():
            if mn in rd:
                v = rd[mn]
                if PANDAS_AVAILABLE and pd.isna(v): continue
                m[fn] = self._convert_value(v, fn)
        m = self._fill_missing_fields(m)
        m, _ = self._validate_and_correct(m)
        return m

    def _parse_contact_row_horizontal(self, rd, col_map):
        return self._parse_social_contact_row_horizontal(rd, col_map)

    def _parse_social_contact_row_horizontal(self, rd, col_map):
        c = {}
        for fn, mn in col_map.items():
            if mn in rd:
                v = rd[mn]
                if PANDAS_AVAILABLE and pd.isna(v): continue
                c[fn] = self._convert_value(v, fn)
        c = self._fill_missing_fields(c)
        c, _ = self._validate_and_correct(c)
        return c

    # --- Column mapping ---

    def _normalize_column_name(self, name):
        if not name: return ''
        name = str(name).strip().lower()
        name = re.sub(r'[\s\-_]+', '_', name)
        name = re.sub(r'[^\w]', '', name)
        return name

    def _smart_column_match(self, headers):
        cm = {}
        col_mapping = getattr(self, 'COLUMN_MAPPING', {})
        for idx, h in enumerate(headers):
            nh = self._normalize_column_name(h)
            for sn, aliases in col_mapping.items():
                if nh in [self._normalize_column_name(a) for a in aliases]:
                    cm[sn] = idx
                    break
        return cm

    def _dynamic_column_match(self, headers, target_fields):
        result = {}
        if dynamic_column_match is not None:
            for tf in target_fields:
                m = dynamic_column_match(headers, tf)
                if m: result[tf] = m
        return result

    def _classify_sheet(self, headers, rows):
        fk = ['member','family','relationship','家属','家庭','成员','关系']
        sk = ['contact','social','接触','社会','社交']
        hs = ' '.join(str(h).lower() for h in headers)
        fs = sum(1 for kw in fk if kw in hs)
        ss = sum(1 for kw in sk if kw in hs)
        if fs > ss: return 'family'
        if ss > fs: return 'social'
        return 'mixed'

    def _classify_sheet_type_from_rows(self, rows):
        if not rows: return 'mixed'
        fi, si = 0, 0
        for row in rows[:min(20,len(rows))]:
            if not row: continue
            rs = ' '.join(str(c) for c in row).lower()
            if any(kw in rs for kw in ['配偶','子女','父母','关系']): fi += 1
            if any(kw in rs for kw in ['同事','朋友','邻居']): si += 1
        if fi > si: return 'family'
        if si > fi: return 'social'
        return 'mixed'

    # --- Encoding & header ---

    def _detect_file_encoding(self, file_path):
        for enc in ['utf-8-sig','utf-8','gbk','gb2312','gb18030','latin-1']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    f.read(1024)
                return enc
            except (UnicodeDecodeError, UnicodeError):
                continue
        return 'utf-8'

    def _detect_header_row(self, rows):
        if not rows: return None
        hk = ['name','姓名','age','年龄','member','contact','relationship','关系','duration','时长']
        for idx, row in enumerate(rows[:min(10,len(rows))]):
            rs = ' '.join(str(c).lower() for c in row if c)
            if any(kw in rs for kw in hk): return idx
        return 0

    def _merge_multirow_header(self, rows):
        if not rows: return []
        if len(rows) == 1:
            return [str(c).strip() if c else '' for c in rows[0]]
        mc = max(len(r) for r in rows)
        merged = []
        for ci in range(mc):
            parts = []
            for r in rows:
                if ci < len(r) and r[ci]:
                    parts.append(str(r[ci]).strip())
            merged.append(' '.join(parts) if parts else f'col_{ci}')
        return merged

    # --- Value conversion ---

    def _convert_value(self, value, field_name):
        if value is None or (isinstance(value, str) and value.strip() == ''):
            return DEFAULT_VALUES.get(field_name, MISSING_VALUE_MARKER)
        if field_name in BOOLEAN_FIELDS:
            return self._safe_bool(value)
        if field_name in CLINICAL_FIELDS:
            if isinstance(value, str) and value.strip() in CLINICAL_TEXT_MAP:
                return CLINICAL_TEXT_MAP[value.strip()]
            return self._safe_numeric(value, 0)
        if field_name in RATING_FIELDS:
            if isinstance(value, str) and value.strip() in RATING_TEXT_MAP:
                return RATING_TEXT_MAP[value.strip()]
            return self._safe_numeric(value, 3)
        if field_name in NUMERIC_FIELDS:
            return self._safe_numeric(value, DEFAULT_VALUES.get(field_name, 0))
        if field_name == 'contact_distance':
            if isinstance(value, str) and value.strip() in DISTANCE_TEXT_MAP:
                return DISTANCE_TEXT_MAP[value.strip()]
            return str(value).strip() if value else 'medium'
        if field_name == 'exposure_setting':
            if isinstance(value, str) and value.strip() in SETTING_TEXT_MAP:
                return SETTING_TEXT_MAP[value.strip()]
            return str(value).strip() if value else 'general'
        if isinstance(value, (int, float)):
            return value
        return str(value).strip() if value else ''

    def _fill_missing_fields(self, record):
        if not record: return {}
        for field, default in DEFAULT_VALUES.items():
            if field not in record or record[field] is None:
                record[field] = default
        return record

    def _validate_and_correct(self, record):
        warnings = []
        if not record: return record, warnings
        for field, value in list(record.items()):
            if field in FIELD_RANGES:
                rng = FIELD_RANGES[field]
                try:
                    val = float(value) if value is not None else rng.get('min', 0)
                    mn = rng.get('min', 0)
                    mx = rng.get('max', float('inf'))
                    if val < mn:
                        warnings.append(f"{field} below min {mn}, corrected")
                        record[field] = mn
                    elif val > mx:
                        warnings.append(f"{field} above max {mx}, corrected")
                        record[field] = mx
                except (ValueError, TypeError):
                    record[field] = DEFAULT_VALUES.get(field, 0)
        return record, warnings

    def _safe_numeric(self, value, default=0):
        if value is None: return default
        try:
            if isinstance(value, (int, float)): return value
            if isinstance(value, str):
                value = value.strip()
                if not value: return default
                if '.' in value: return float(value)
                return int(value)
            return int(value)
        except (ValueError, TypeError):
            return default

    def _safe_bool(self, value, default=0):
        if value is None: return default
        if isinstance(value, bool): return 1 if value else 0
        if isinstance(value, (int, float)): return 1 if value else 0
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ('是','有','已接种','曾患','阳性','1','yes','true','y','真'): return 1
            if v in ('否','无','未接种','未患','阴性','0','no','false','n','假'): return 0
        return default

    def _infer_missing_values(self, records):
        if not records: return records
        for record in records:
            if hasattr(self, '_apply_scenario_defaults'):
                record = self._apply_scenario_defaults([record], 'custom')[0]
            record = self._fill_missing_fields(record)
        return records

    def _parse_date_columns(self, records):
        for record in records:
            for key, value in list(record.items()):
                if isinstance(value, str) and ('date' in key.lower() or 'time' in key.lower()):
                    try:
                        for fmt in ['%Y-%m-%d','%Y/%m/%d','%d/%m/%Y','%m/%d/%Y']:
                            try:
                                record[key] = datetime.datetime.strptime(value, fmt)
                                break
                            except ValueError: continue
                    except Exception as _e:
                        LOGGER.debug("解析日期列失败 key=%s value=%r: %s", key, value, _e)
        return records

    def _apply_scenario_defaults(self, records, scenario):
        sd = {
            'oilfield': {'exposure_setting':'oilfield_camp','ventilation':3},
            'hospital': {'exposure_setting':'closed','ventilation':2},
            'school': {'exposure_setting':'crowded','ventilation':3},
            'custom': {},
        }
        defaults = sd.get(scenario, {})
        if not defaults: return records
        for record in records:
            for key, val in defaults.items():
                if key not in record or record[key] is None or record[key] == MISSING_VALUE_MARKER:
                    record[key] = val
        return records

    # --- Reports ---

    def generate_report(self, format='html', ai_enhanced=False, ai_config=None):
        # 问题七：委托 export_utils.generate_assessment_report，
        # 消除与 export_utils/report_export.py 的职责重叠。
        # 缺陷 #14：透传 ai_enhanced / ai_config 参数，支持 AI 增强报告。
        try:
            from ..export_utils import generate_assessment_report as _gen
            results = {
                'family_members': getattr(self, 'family_members', []),
                'social_contacts': getattr(self, 'social_contacts', []),
                'patient_info': getattr(self, 'patient_info', {}),
                'results': getattr(self, 'results', {}),
            }
            return _gen(results, format=format, ai_enhanced=ai_enhanced,
                        ai_config=ai_config)
        except ImportError:
            LOGGER.warning("export_utils not available")
            return ''
        except Exception as e:
            LOGGER.warning("Report generation failed: %s", e, exc_info=True)
            return ''

    def export_model_training_report(self, predictor, filepath, format='html'):
        try:
            from ..export_utils import export_model_training_report as _etr
            return _etr(predictor, filepath, format=format)
        except ImportError:
            report = f"""<!DOCTYPE html><html lang="zh-CN">
<head><meta charset="UTF-8"><title>Training Report</title></head>
<body><h1>Model Training Report</h1>
<p>Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p>Trained: {getattr(predictor,'is_trained',False)}</p>
<p>Samples: {getattr(predictor,'training_sample_count',0)}</p>
<p>Real Data: {getattr(predictor,'use_real_data',False)}</p>
</body></html>"""
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(report)
                return True
            except Exception as e:
                LOGGER.warning("Training report export failed: %s", e)
                return False
        except Exception as e:
            LOGGER.warning("Model training report export failed: %s", e, exc_info=True)
            return False

    # --- Info ---

    def get_import_errors(self):
        return getattr(self, 'csv_import_errors', [])

    def get_import_summary(self):
        return {
            'family_members': len(getattr(self, 'family_members', [])),
            'social_contacts': len(getattr(self, 'social_contacts', [])),
            'errors': len(getattr(self, 'csv_import_errors', [])),
        }

    def _add_import_error(self, msg):
        if not hasattr(self, 'csv_import_errors'):
            self.csv_import_errors = []
        self.csv_import_errors.append(msg)

    # --- Dedup ---

    def _merge_with_deduplication(self, existing, new):
        if not existing: return list(new)
        if not new: return list(existing)
        merged = list(existing)
        en = set()
        for r in existing:
            n = r.get('member_name') or r.get('contact_name') or ''
            if n: en.add(n.strip().lower())
        for r in new:
            n = r.get('member_name') or r.get('contact_name') or ''
            if n and n.strip().lower() in en: continue
            merged.append(r)
        return merged

