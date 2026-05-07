from enum import StrEnum

class PrimaryClass(StrEnum):
    MEDICAL = "medical"
    NON_MEDICAL = "non_medical"

class Subcategory(StrEnum):
    LAB = "lab"
    CLI = "clinical_document"
    HEALTH_CHECK = "health_check"
    IMAGING_REPORT = "imaging_report"
    IPD_OPD_DOCUMENT = "ipd_opd_document"
    MEDICAL_CERTIFICATE = "medical_certificate"
    DISCHARGE_SUMMARY = "discharge_summary"
    MEDICAL_OTHER = "medical_other"
    PASSPORT = "passport"
    ID = "id"
    FINANCIAL = "financial"
    OTHER = "other"

VALID_SUBCATEGORIES: dict[PrimaryClass, set[Subcategory]] = {
    PrimaryClass.MEDICAL: {
        Subcategory.LAB,
        Subcategory.HEALTH_CHECK,
        Subcategory.IMAGING_REPORT,
        Subcategory.IPD_OPD_DOCUMENT,
        Subcategory.MEDICAL_CERTIFICATE,
        Subcategory.DISCHARGE_SUMMARY,
        Subcategory.CLI,
        Subcategory.MEDICAL_OTHER,
    },
    PrimaryClass.NON_MEDICAL: {
        Subcategory.PASSPORT,
        Subcategory.ID,
        Subcategory.FINANCIAL,
        Subcategory.OTHER,
    },
}

class ClassificationCode(StrEnum):
    MED = "MED"
    NON = "NON"
    LAB = "LAB"
    CHK = "CHK"
    CLI = "CLI"
    PS = "PS"
    ID = "ID"
    FIN = "FIN"
    OTH = "OTH"

CODE_TO_PRIMARY: dict[ClassificationCode, PrimaryClass] = {
    ClassificationCode.MED: PrimaryClass.MEDICAL,
    ClassificationCode.NON: PrimaryClass.NON_MEDICAL,
}

CODE_TO_SUBCATEGORY: dict[ClassificationCode, Subcategory] = {
    ClassificationCode.LAB: Subcategory.LAB,
    ClassificationCode.CHK: Subcategory.HEALTH_CHECK,
    ClassificationCode.CLI: Subcategory.CLI,
    ClassificationCode.PS: Subcategory.PASSPORT,
    ClassificationCode.ID: Subcategory.ID,
    ClassificationCode.FIN: Subcategory.FINANCIAL,
    ClassificationCode.OTH: Subcategory.OTHER,
}

SUBCATEGORY_TO_CODE: dict[Subcategory, ClassificationCode] = {
    v: k for k, v in CODE_TO_SUBCATEGORY.items()
}

PRIMARY_TO_CODE: dict[PrimaryClass, ClassificationCode] = {
    v: k for k, v in CODE_TO_PRIMARY.items()
}

VALID_ROOT_CODES: set[ClassificationCode] = {
    ClassificationCode.MED, ClassificationCode.NON,
}
VALID_MED_SUB_CODES: set[ClassificationCode] = {
    ClassificationCode.LAB,
    ClassificationCode.CHK,
    ClassificationCode.CLI,
    ClassificationCode.OTH,
}
VALID_NONMED_SUB_CODES: set[ClassificationCode] = {
    ClassificationCode.PS,
    ClassificationCode.ID,
    ClassificationCode.FIN,
    ClassificationCode.OTH,
}
