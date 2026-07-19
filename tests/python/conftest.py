from __future__ import annotations

import pytest


@pytest.fixture
def kb(tmp_path):
    import ragvault

    knowledge_base = ragvault.open(tmp_path / "kb")
    yield knowledge_base
    knowledge_base.close()


@pytest.fixture
def docs_dir(tmp_path):
    d = tmp_path / "documents"
    d.mkdir()
    (d / "cancellation.md").write_text(
        "# Cancellation Policy\n\n"
        "## Refunds\n\nCustomers may cancel within 30 days for a full refund.\n\n"
        "## Exceptions\n\nDigital goods already downloaded are not refundable.\n"
    )
    (d / "shipping.md").write_text(
        "# Shipping\n\nOrders ship within 5 business days nationwide.\n"
    )
    (d / "accounts.txt").write_text(
        "Accounts can be closed at any time by the user from the settings page.\n"
    )
    return d
