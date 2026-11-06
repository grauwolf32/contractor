/* Host-owned design interaction. No real credential is created or retained. */
export function openCredentialDialog({kind, kindLabel, credentialKind, credentials, onCreated}) {
  if (document.querySelector('#credential-dialog')) return;
  const opener = document.activeElement;
  const dialog = document.createElement('dialog');
  dialog.id = 'credential-dialog';
  dialog.className = 'eg-credential-dialog';
  dialog.setAttribute('aria-labelledby', 'credential-title');
  dialog.setAttribute('aria-describedby', 'credential-demo-note');
  const field = (label, id, type = 'text') => `<div class="eg-field"><label class="eg-label" for="${id}">${label}</label><input id="${id}" type="${type}" autocomplete="${type === 'password' ? 'new-password' : 'off'}" aria-describedby="${id}-error"><span class="eg-field-error" id="${id}-error"></span></div>`;
  dialog.innerHTML = `<form novalidate autocomplete="off">
    <header class="eg-credential-heading"><span class="eg-section-icon" aria-hidden="true">+</span><div><span class="eg-eyebrow">${kindLabel} · demo</span><h2 id="credential-title">Add credential</h2></div><button class="eg-btn" data-quiet type="button" id="credential-close" aria-label="Close credential form">×</button></header>
    <p id="credential-demo-note" class="eg-note">Use test values only. This preview adds temporary metadata and sends no API request. Reloading clears added credentials.</p>
    ${field('Credential ID', 'credential-id')}
    <p class="eg-help">A unique reference for this ${kindLabel} identity. The policy contains this ID, never the secret.</p>
    <div class="eg-credential-material">${kind === 'basic' ? `<div class="eg-grid">${field('Username · write only', 'credential-username')}${field('Password · write only', 'credential-password', 'password')}</div>` : kind === 'bearer' ? field('Bearer token · write only', 'credential-token', 'password') : '<div id="credential-cookie-rows"></div><button class="eg-btn" type="button" id="credential-add-cookie">+ Add cookie</button>'}</div>
    <p class="eg-help">Entered values are cleared on submit or close. Full validation remains with the existing credential operation.</p>
    <div class="eg-credential-error" role="alert" id="credential-error" hidden></div>
    <footer class="eg-credential-actions"><button class="eg-btn" type="button" id="credential-cancel">Cancel</button><button class="eg-btn" data-primary type="submit">Create &amp; use · demo</button></footer>
  </form>`;
  document.body.append(dialog);
  const find = id => dialog.querySelector('#' + id);
  const form = dialog.querySelector('form');
  const idSchema = window.CREDENTIAL_SCHEMA.$defs.RuntimeCredentialId;
  find('credential-id').maxLength = idSchema.maxLength;
  if (kind === 'basic') {
    find('credential-username').maxLength = window.CREDENTIAL_SCHEMA.$defs.BasicCredentialUsername.maxLength;
    find('credential-password').maxLength = window.CREDENTIAL_SCHEMA.$defs.BasicCredentialPassword.maxLength;
  } else if (kind === 'bearer') find('credential-token').maxLength = window.CREDENTIAL_SCHEMA.$defs.TargetBearerMaterial.properties.token.maxLength;
  let nextCookie = 0;
  const cookieSchema = window.CREDENTIAL_SCHEMA.$defs.TargetCookiesMaterial.properties.cookies;
  function addCookie() {
    const row = document.createElement('div');
    const index = ++nextCookie;
    row.className = 'eg-cookie-row';
    row.dataset.cookieRow = '';
    row.innerHTML = `${field('Cookie name', `credential-cookie-name-${index}`)}${field('Cookie value · write only', `credential-cookie-value-${index}`, 'password')}<button class="eg-btn" data-quiet data-danger type="button" aria-label="Remove cookie ${index}">Remove</button>`;
    find('credential-cookie-rows').append(row);
    if (Number.isSafeInteger(cookieSchema.propertyNames.maxLength)) row.querySelector('input').maxLength = cookieSchema.propertyNames.maxLength;
    row.querySelector('input[type="password"]').maxLength = cookieSchema.additionalProperties.maxLength;
    row.querySelector('button').onclick = () => {
      row.querySelectorAll('input').forEach(input => { input.value = ''; });
      row.remove();
      updateCookieActions();
      find('credential-add-cookie').focus();
    };
    updateCookieActions();
    return row.querySelector('input');
  }
  function updateCookieActions() {
    const rows = dialog.querySelectorAll('[data-cookie-row]');
    rows.forEach(row => { row.querySelector('button').disabled = rows.length === 1; });
    find('credential-add-cookie').disabled = rows.length >= cookieSchema.maxProperties;
  }
  if (kind === 'cookies') {
    addCookie();
    find('credential-add-cookie').onclick = () => addCookie().focus();
  }
  function clearMaterial() {
    dialog.querySelectorAll('.eg-credential-material input').forEach(input => { input.value = ''; });
  }
  function dismiss() {
    clearMaterial();
    dialog.close();
    dialog.remove();
    if (opener?.isConnected) opener.focus();
  }
  ['credential-close', 'credential-cancel'].forEach(id => { find(id).onclick = dismiss; });
  dialog.addEventListener('cancel', event => { event.preventDefault(); dismiss(); });
  form.onsubmit = event => {
    event.preventDefault();
    const credentialId = find('credential-id').value.trim();
    let material;
    let duplicateCookie = false;
    if (kind === 'basic') material = {username: find('credential-username').value, password: find('credential-password').value};
    else if (kind === 'bearer') material = {token: find('credential-token').value};
    else {
      const entries = [...dialog.querySelectorAll('[data-cookie-row]')].map(row => [...row.querySelectorAll('input')].map(input => input.value));
      duplicateCookie = new Set(entries.map(([name]) => name)).size !== entries.length;
      material = {cookies: Object.fromEntries(entries)};
    }
    const valid = window.CREDENTIAL_SCHEMA_VALIDATE({credentialId, kind: credentialKind, material}).valid;
    const idValid = new RegExp(idSchema.pattern).test(credentialId) && credentialId.length <= idSchema.maxLength;
    const duplicateId = credentials.some(credential => credential.id === credentialId);
    // Retain a verdict only; material never enters metadata, policy state or logs.
    material = undefined;
    clearMaterial();
    dialog.querySelectorAll('[aria-invalid]').forEach(input => input.removeAttribute('aria-invalid'));
    find('credential-id-error').textContent = '';
    let message = '';
    if (!idValid) message = 'Use a lowercase credential ID starting with a letter; only letters, numbers, underscores and hyphens are allowed.';
    else if (duplicateId) message = 'This credential ID already exists. Choose a new ID.';
    else if (credentials.length >= 200) message = 'This preview catalog is full.';
    else if (duplicateCookie) message = 'Cookie names must be unique. Re-enter test values to try again.';
    else if (!valid) message = `Check the ${kindLabel} fields and re-enter test values. Values have been cleared.`;
    find('credential-error').hidden = !message;
    find('credential-error').textContent = message;
    if (message) {
      if (!idValid || duplicateId) {
        find('credential-id').setAttribute('aria-invalid', 'true');
        find('credential-id-error').textContent = message;
        find('credential-id').focus();
      } else dialog.querySelector('.eg-credential-material input').focus();
      return;
    }
    dismiss();
    onCreated({id: credentialId, kind: credentialKind, label: kindLabel + ' · demo'});
  };
  dialog.showModal();
  find('credential-id').focus();
}
