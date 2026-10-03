"""Observable monomer-product view for measured singly charged dimer dissociation."""

from casmi_ml.metfrag_dimer import monomer_product_row


def input_view(row):
    transformed = monomer_product_row(row)
    if transformed is None:
        return row, False
    # Preserve acquisition metadata while replacing only the supported ion view.
    return dict(row, **transformed), True
